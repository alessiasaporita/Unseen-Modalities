from ast_model import ASTModel
from dataloader_train import EPICKitchensTrain
from dataloader_validation import EPICKitchensValidation
import torch
import argparse
import tqdm
import os
import shutil
import numpy as np
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from ast_configs import get_audio_configs
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
from vit import ViT
from feature_reorganization import ViTReorganization
import wandb
import os
from torch.optim.lr_scheduler import MultiStepLR

"""
    https://github.com/gerasmark/Reproducing-Unseen-Modality-Interaction/blob/main/main.ipynb
"""

def dict_to_cuda(data):
    for key, value in data.items():
        data[key] = value.cuda()
    return data

def spatialtemporal2tokens(data):
    b, c, f, h, w = data.size()
    data = data.view(b, c, f * h * w)
    data = data.transpose(1, 2).contiguous()
    return data

"""
def save_pseudo_labels(outputs, keys, masks):
    truth_outputs = outputs[:,0,:] #(B, 3086) = first CLS token = predictions
    detached_outputs = truth_outputs.detach().cpu()
    detached_outputs = torch.softmax(detached_outputs, dim=-1)

    #For each sample in the batch, save its relative prediction
    for i in range(len(keys)): 
        #------------RGB------------
        if masks['RGB'][i].sum()!=0: #RGB sample
            save_path = "rgb_pseudo/{}.npy".format(keys[i])
            if os.path.exists(save_path): #predictions for i-th sample 
                rgb_pseudo = np.load(save_path)
                if rgb_pseudo.shape[0]>=10:
                    rgb_pseudo=rgb_pseudo[-9:]
                rgb_pseudo = np.concatenate((rgb_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                rgb_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("rgb_pseudo/{}.npy".format(keys[i]), rgb_pseudo)
        #------------Audio------------
        if masks['Audio'][i].sum()!=0: #Audio sample
            save_path = "audio_pseudo/{}.npy".format(keys[i])
            if os.path.exists(save_path):
                audio_pseudo = np.load(save_path)
                if audio_pseudo.shape[0]>=10:
                    audio_pseudo=audio_pseudo[-9:]
                audio_pseudo = np.concatenate((audio_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                audio_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("audio_pseudo/{}.npy".format(keys[i]), audio_pseudo)

"""
class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing #0.1

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1) #class probabilities
        #tensor of the same shape as the input with values equal to self.smoothing / (input.size(-1) - 1.0)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        #For the target class, it sets the value to 1.0 - self.smoothing.
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean() #cross-entropy loss
        return loss

class AlignmentModule(nn.Module):
    def __init__(self, dim=256):
        super(AlignmentModule, self).__init__()
        self.base_vectors = nn.Parameter(torch.randn(1, 3806, dim)) #(1, 3086, d*), d*=256, class tokens to be learnt 
    
    def forward(self, input): # input [B, 512, 256]
        input = torch.mean(input, dim=1, keepdim=True) # [B, 1, d*], mean vectors of the samples in the batch
        base_vectors = self.base_vectors.repeat(input.size()[0], 1, 1) #[B, 3806, d*], class tokens
        sim = torch.mean((base_vectors - input) ** 2, dim=-1) #[B, 3806], euclidean distance between the average vectors of the batch and each class tokens
        return sim 

def extract_features(unimodal_models, data):
    outputs = {}
    for key, value in data.items(): #key = Audio, RGB
        outputs[key] = unimodal_models[key](value)
        if key == 'RGB':
            outputs[key] = spatialtemporal2tokens(outputs[key])
    return outputs

def train_one_step(
    data,
    labels,
    masks,
    audio_pseudo,
    rgb_pseudo,
    unimodal_models,
    multimodal_model,
    reorganization_module,
    alignment_model,
    optim,
    loss_fn,
    kl_loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
    deactivate_KL,
):

    with torch.no_grad():
        outputs = extract_features(unimodal_models, data) #RGB = (B, 784, 768), Audio = (B, 146, 768)

    rgb, audio = reorganization_module( #feature projection = (B, 512, 256) = (B, k*, d*)
        outputs['RGB'], outputs['Audio'] 
    ) 

    audio_sim = alignment_model(audio) #sim = [B, 3806]
    rgb_sim = alignment_model(rgb) #sim = [B, 3806]

    outputs = multimodal_model( #(B, 2, 3086), the two CLS tokens
        rgb, audio, masks['RGB'], masks['Audio']
    ) 

    #Audio and RGB sample indices
    audio_indices = torch.sum(masks['Audio'].squeeze(-1), dim=-1) > 0 #(B,) 
    rgb_indices = torch.sum(masks['RGB'].squeeze(-1), dim=-1) > 0 #(B,) 

    #ALIGNMENT LOSS: max similarity between average vectors of the samples and the corrisponding class tokens, ie min euclidean distance between average vectors and the class tokens
    #Audio and RGB labels
    audio_labels = labels[audio_indices] 
    rgb_labels = labels[rgb_indices] 
    audio_onehot_labels = F.one_hot(audio_labels, num_classes = 3806) 
    rgb_onehot_labels = F.one_hot(rgb_labels, num_classes = 3806) 
    
    #Audio and RGB distances
    audio_sim = audio_sim[audio_indices] 
    rgb_sim = rgb_sim[rgb_indices] 
    audio_sim = torch.sum(audio_sim * audio_onehot_labels, dim=-1) 
    rgb_sim = torch.sum(rgb_sim * rgb_onehot_labels, dim=-1) 

    alignment_loss = (torch.sum(audio_sim) + torch.sum(rgb_sim)) / (torch.sum(audio_indices) + torch.sum(rgb_indices))

    #Total Loss: L-supervised + gamma L-pseudo + alpha L-align, with gamma = 3000, alpha = 0.001 

    if deactivate_KL: 
        output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 +  0.001 * alignment_loss
    else:
        #PSEUDO LOSS: mean KL-divergence between log-prob of audio and pseudo label, rgb and rgb pseudo label, multiplied for their weigths=number of rgb/audio samples
        audio_pseudo = audio_pseudo[audio_indices]
        rgb_pseudo = rgb_pseudo[rgb_indices]
        probs = torch.softmax(outputs[:,1], dim=-1)
        audio_prob = probs[audio_indices] 
        rgb_prob = probs[rgb_indices] 
        kl_loss = torch.mean(kl_loss_fn(torch.log(audio_prob), audio_pseudo)) * torch.sum(audio_indices) + torch.mean(kl_loss_fn(torch.log(rgb_prob), rgb_pseudo)) * torch.sum(rgb_indices)
        
        #Total loss
        output_loss = loss = loss_fn(outputs[:,0], labels) + kl_loss / labels.size()[0] * 3000 +  0.001 * alignment_loss
    
    loss = loss / gc
    scaler.scale(loss).backward()

    if((indice + 1) % gc == 0) or (indice + 1 == last_indice):
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return outputs, output_loss


def val_one_step(
    data,
    labels,
    masks,
    unimodal_models,
    multimodal_model,
    reorganization_module,
    alignment_model,
    loss_fn,
    gc,
):
    with torch.no_grad():
        outputs = extract_features(unimodal_models, data)

        rgb, audio = reorganization_module(
            outputs['RGB'], outputs['Audio']
        )
        outputs = multimodal_model(
            rgb, audio, masks['RGB'], masks['Audio']
        )
        output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5
        #loss = loss / gc
    
    return outputs, output_loss


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=1e-1
    )  
    parser.add_argument("--batch_size", type=int, help="batch size", default=96)
    parser.add_argument("--deactivate_KL", type=bool, help="Deactivate KL loss", default=False)
    parser.add_argument(
        "--audio_data_path",
        type=str,
        help="path to data",
        default="/path/to/EPIC-KITCHENS-Audio-Clip/",
    )
    parser.add_argument(
        "--rgb_data_path",
        type=str,
        help="path to data",
        default="/path/to/EPIC-KITCHENS/",
    )
    parser.add_argument(
        "--save_name", type=str, help="name to save the model", default="1e-1",
    ) #1e-1, ...
    parser.add_argument(
        "--resume_training", type=bool, help="resume training or not", default=False
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
    )
    parser.add_argument(
        "--save_all", type=bool, help="save all checkpoints or not", default=False
    )
    parser.add_argument(
        "--num_position", type=int, help="number of projection tokens", default=512,
    )
    parser.add_argument(
        "--workers", type=int, help="number of workers", default=16,
    )
    parser.add_argument(
        "--num_epochs", type=int, help="number of epochs", default=120,
    )
    parser.add_argument(
        "--gc", type=int, help="gradient accumulation", default=2,
    )
    args = parser.parse_args()

    wandb.init(
        project="Unseen_Modalities Baseline",
        name='Unseen Modalities',
        config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gc": args.gc,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_training": args.resume_training,
        "deactivate_KL": args.deactivate_KL,
        }
    )

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    device = "cuda"  # or 'cpu'
    device = torch.device(device)

    base_path = "/work/tesi_asaporita/UnseenModalities/checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    batch_size = args.batch_size #96
    target_length = 128
    train_audio_configs, val_audio_configs = get_audio_configs(
        target_length=target_length
    )

    """
    Pretrained unimodal encoders
    Audio: AST
    RGB: SWIN-T
    """
    #audio
    audio_model = ASTModel( 
        label_dim=3806,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=target_length, #128
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
    ) 
    audio_model = audio_model.to(device)
    audio_model = nn.DataParallel(audio_model)
    checkpoint = torch.load("/work/tesi_asaporita/UnseenModalities/audio3/checkpoints/audio7.pt")
    audio_model.load_state_dict(checkpoint["model"])
    audio_model.eval() 

    #rgb
    rgb_model = omnivore_swinT(pretrained=True) 
    rgb_model.heads = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
    )
    rgb_model.multimodal_model = False
    rgb_model = torch.nn.DataParallel(rgb_model)
    checkpoint = torch.load("/work/tesi_asaporita/UnseenModalities/checkpoints/best_unimodal_rgb65.p")
    rgb_model.load_state_dict(checkpoint["model"]) 
    rgb_model = rgb_model.to(device)
    rgb_model.eval()

    unimodal_models = { #audio, RGB
        'RGB': rgb_model,
        'Audio': audio_model
    }

    """
    Multimodal Transfomer
    """
    multimodal_model = ViT( #Total number of trainable parameters: 5.841.630 -> 6.239.454 = 6M
        num_classes=3806,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position,
    )
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device)

    """
    Feature projection: project unimodal embeddings into a common feature space (K^m, d^m)-dim
    """
    reorganization_module = ViTReorganization( #Total number of trainable parameters: 10.759.680 -> 10.361.856 = 10M
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position, #512
    )
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device)

    """
    Alignment: align embeddings with learnable class tokens 
    """
    alignment_model = AlignmentModule() #Total number of trainable parameters: 974.336 = 0.97M
    alignment_model = torch.nn.DataParallel(alignment_model)
    alignment_model = alignment_model.to(device)

    loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    kl_loss_fn = nn.KLDivLoss(reduce=False) #loss pseudolabel
    kl_loss_fn = kl_loss_fn.cuda()

    optim = torch.optim.SGD(
        list(multimodal_model.parameters())+list(reorganization_module.parameters()) + list(alignment_model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = MultiStepLR(optim, milestones=[70], gamma=0.1)
    scaler = GradScaler()
    BestLoss = float("inf")
    initial_epoch = 0
    BestEpoch = 0
    BestAcc = 0

    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        multimodal_model.load_state_dict(checkpoint["model"])
        reorganization_module.load_state_dict(checkpoint["reorganization"])
        alignment_model.load_state_dict(checkpoint["alignment"])
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) 
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_epoch = checkpoint['epoch'] + 1
        BestLoss = checkpoint['best_loss']
        BestAcc = checkpoint['best_acc']


    train_loader = torch.utils.data.DataLoader(
        EPICKitchensTrain(
            audio_conf=train_audio_configs,
            split="train",
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        EPICKitchensValidation(
            audio_conf=val_audio_configs,
            split="validation",
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    log_path = "logs/{}.csv".format(args.save_name)
    
    print("---------------Start Training---------------")
    with open(log_path, "a") as f:
        for epoch_i in range(initial_epoch, args.num_epochs):
            print("Epoch: %02d" % epoch_i)
            for split in ["train", "val"]:
                acc = 0
                count = 0
                total_loss = 0
                loss = 0
                print(split)
                multimodal_model.train(split == "train")
                reorganization_module.train(split == "train")  
                alignment_model.train(split == "train")  

                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i,(data, labels, masks, audio_pseudo, rgb_pseudo, keys),) in enumerate(dataloaders[split]):
                        data = dict_to_cuda(data) #dict with RGB =(B, 3, 32, 224, 224), Audio=(B, 128, 128)
                        masks = dict_to_cuda(masks) #dict with 'RGB'=(B, 512, 1), Audio=(B, 512, 1)
                        labels = labels.cuda() #(B, )
                        audio_pseudo = audio_pseudo.cuda() #(3086, )
                        rgb_pseudo = rgb_pseudo.cuda() #(3086, )

                        if split == "train": 
                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                audio_pseudo,
                                rgb_pseudo,
                                unimodal_models,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                optim,
                                loss_fn,
                                kl_loss_fn,
                                scaler,
                                i,
                                len(dataloaders[split]),
                                args.gc,
                                args.deactivate_KL,
                            )


                        else:  #val
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
                                unimodal_models,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                loss_fn,
                                args.gc,
                            )
                        
                        wandb.log({"{}/step_loss".format(split): loss}) #step loss
                        total_loss += loss.item() * batch_size

                        outputs = torch.softmax(outputs, dim=-1) #(B, 2, 3086)
                        outputs = torch.mean(outputs, dim=1) #(B, 1, 3086) = mean of the predictions of the two CLS tokens 
                        _, predict = torch.max(outputs, dim=1)
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)

                        count += outputs.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(
                                total_loss / float(count),
                                loss.item(),
                                acc / float(count),
                            )
                        )
                        pbar.update()
                    f.write(
                        "{},{},{},{}\n".format(
                            epoch_i,
                            split,
                            total_loss / float(count),
                            acc / float(count),
                        )
                    )
                    f.flush()
                    #wandb log, split = train, val
                    wandb.log({"{}/loss".format(split): total_loss / float(count), "{}/loss_epoch".format(split): epoch_i}) #epoch loss
                    wandb.log({"{}/acc".format(split): acc / float(count), "{}/acc_epoch".format(split): epoch_i}) #epoch accuracy 

            scheduler.step()
            wandb.log({"train/lr": scheduler.get_last_lr()[0]}) #epoch lr

            if acc / float(count) > BestAcc: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestAcc = acc / float(count)
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
                    "alignment": alignment_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": BestLoss,
                    "best_acc": BestAcc,
                }

                torch.save(
                    save, base_path + "best_multimodal{}{}.pt".format(args.save_name, epoch_i)
                ) 
            if args.save_all and epoch_i % 4 == 0: #save model every 4 epochs
                save = {
                    "epoch": epoch_i,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": BestLoss,
                    "best_acc": BestAcc,
                }

                torch.save(
                    save, base_path + "best_unimodal{}{}.pt".format(args.save_name, epoch_i)
                )    
    f.close()
