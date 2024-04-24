import torch
import argparse
import tqdm
import os
import webdataset as wds
import shutil
import numpy as np
import torch.nn as nn
import random
from braceexpand import braceexpand
import warnings
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
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

def save_pseudo_labels(outputs, keys, masks):
    truth_outputs = outputs[:,0,:] #(B, 3086) = first CLS token = predictions
    detached_outputs = truth_outputs.detach().cpu()
    detached_outputs = torch.softmax(detached_outputs, dim=-1)

    #For each sample in the batch, save its relative prediction
    for i in range(len(keys)): 
        #------------RGB------------
        if masks['RGB'][i].sum()!=0: #RGB sample
            save_path = "rgb_pseudo_web/{}.npy".format(keys[i])
            if os.path.exists(save_path): #predictions for i-th sample 
                rgb_pseudo = np.load(save_path)
                if rgb_pseudo.shape[0]>=10:
                    rgb_pseudo=rgb_pseudo[-9:]
                rgb_pseudo = np.concatenate((rgb_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                rgb_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("rgb_pseudo_web/{}.npy".format(keys[i]), rgb_pseudo)
        #------------Audio------------
        if masks['Audio'][i].sum()!=0: #Audio sample
            save_path = "audio_pseudo_web/{}.npy".format(keys[i])
            if os.path.exists(save_path):
                audio_pseudo = np.load(save_path)
                if audio_pseudo.shape[0]>=10:
                    audio_pseudo=audio_pseudo[-9:]
                audio_pseudo = np.concatenate((audio_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                audio_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("audio_pseudo_web/{}.npy".format(keys[i]), audio_pseudo)

def get_pseudo_labels(keys, masks, deactivate_KL):
    audio_pseudo_labels = []
    rgb_pseudo_labels = []
    for i in range(len(keys)): 
        #------------Audio------------
        if masks['Audio'][i].sum()!=0: #Audio sample
            audio_pseudo_path = "rgb_pseudo_web/{}.npy".format(keys[i])
            if deactivate_KL or not os.path.exists(audio_pseudo_path): #for the first epoch, KL always deactive
                audio_pseudo = torch.zeros((3806,))
            else:
                audio_pseudo = torch.Tensor(np.load(audio_pseudo_path))
                audio_pseudo = torch.mean(audio_pseudo, dim=0) #(3086, )
        else: #RGB sample
            audio_pseudo = torch.zeros((3806,))

        #------------RGB------------
        if masks['RGB'][i].sum()!=0: #RGB sample
            rgs_preudo_path = "audio_pseudo_web/{}.npy".format(keys[i])
            if deactivate_KL or not os.path.exists(rgs_preudo_path): #for the first epoch, KL always deactive
                rgb_pseudo = torch.zeros((3806, ))
            else:
                rgb_pseudo = torch.Tensor(np.load(rgs_preudo_path))
                rgb_pseudo = torch.mean(rgb_pseudo, dim=0)
        else: #Audio sample
            rgb_pseudo = torch.zeros((3806, ))
        
        rgb_pseudo_labels.append(rgb_pseudo)
        audio_pseudo_labels.append(audio_pseudo)

    audio_pseudo_labels=torch.stack(audio_pseudo_labels, dim=0)
    rgb_pseudo_labels = torch.stack(rgb_pseudo_labels , dim=0)
    
    return audio_pseudo_labels, rgb_pseudo_labels

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

def train_one_step(
    data,
    labels,
    masks,
    audio_pseudo,
    rgb_pseudo,
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
):
    rgb, audio = reorganization_module( #feature projection = (B, 512, 256) = (B, k*, d*)
        data['RGB'], data['Audio'] 
    ) 

    audio_sim = alignment_model(audio) #sim = [B, 3806]
    rgb_sim = alignment_model(rgb) #sim = [B, 3806]

    outputs = multimodal_model( #(B, 2, 3086), the two CLS tokens
        rgb, audio, masks['RGB'], masks['Audio']
    ) 

    #Audio and RGB sample indices
    audio_indices = torch.sum(masks['Audio'].squeeze(-1), dim=-1) > 0 #(B,) -> indeces of audio samples true/false
    rgb_indices = torch.sum(masks['RGB'].squeeze(-1), dim=-1) > 0 #(B,) -> indeces of RGB samples true/false

    #ALIGNMENT LOSS: max similarity between average vectors of the samples and the corrisponding class tokens, ie min euclidean distance between average vectors and the class tokens
    #Audio and RGB labels
    audio_labels = labels[audio_indices] #(number of audio samples, ), labels of audio samples
    rgb_labels = labels[rgb_indices] #(number of rgb samples, ), labels of rgb samples
    audio_onehot_labels = F.one_hot(audio_labels, num_classes = 3806) #(number of audio samples, 3086) 
    rgb_onehot_labels = F.one_hot(rgb_labels, num_classes = 3806) #(number of rgb samples, 3086) 
    #Audio and RGB distances
    audio_sim = audio_sim[audio_indices] #(number of audio samples, 3086) 
    rgb_sim = rgb_sim[rgb_indices] #(number of RGB samples, 3086)
    audio_sim = torch.sum(audio_sim * audio_onehot_labels, dim=-1) #(number of audio samples, ) 
    rgb_sim = torch.sum(rgb_sim * rgb_onehot_labels, dim=-1) #(number of RGB samples, )

    alignment_loss = (torch.sum(audio_sim) + torch.sum(rgb_sim)) / (torch.sum(audio_indices) + torch.sum(rgb_indices))

    #Total Loss: L-supervised + gamma L-pseudo + alpha L-align, with gamma = 3000, alpha = 0.001 

    if args.deactivate_KL or audio_pseudo.sum()==0 or rgb_pseudo.sum()==0: #if not the first epoch
        output_loss = loss = loss_fn(outputs[:,0], labels) +  0.001 * alignment_loss
    else:
        #PSEUDO LOSS: mean KL-divergence between log-prob of audio and pseudo label, rgb and rgb pseudo label, multiplied for their weigths=number of rgb/audio samples
        audio_pseudo = audio_pseudo[audio_indices]
        rgb_pseudo = rgb_pseudo[rgb_indices]
        probs = torch.softmax(outputs[:,1], dim=-1)
        audio_prob = probs[audio_indices] #(number of audio samples, 3086) 
        rgb_prob = probs[rgb_indices] #(number of rgb samples, 3086) 
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
    multimodal_model,
    reorganization_module,
    alignment_model,
    loss_fn,
    gc,
):
    with torch.no_grad():
        rgb, audio = reorganization_module(
            data['RGB'], data['Audio']
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
        "--train_data_path",
        type=str,
        help="path to train data",
        default="/work/tesi_asaporita/UnseenModalities/webdataset/epic_kitchens-training-{000..012}.tar",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="path to validation data",
        default="/work/tesi_asaporita/UnseenModalities/webdataset/epic_kitchens-validation-{000..001}.tar",
    )
    parser.add_argument(
        "--n_train_samples", type=int, help="number of training samples", default=62297,
    )
    parser.add_argument(
        "--n_val_samples", type=int, help="number of training samples", default=6215,
    )
    parser.add_argument(
        "--save_name", type=str, help="name to save the model", default="1e-1",
    ) 
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
        name='Unseen Modalities WEB',
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

    """
    Multimodal Transfomer
    """
    multimodal_model = ViT(
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
    reorganization_module = ViTReorganization(
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
    alignment_model = AlignmentModule() 
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
    else: #starting training
        if not args.deactivate_KL: #using KL loss
            base_paths = ["audio_pseudo_web/", "rgb_pseudo_web/"]
            for b_path in base_paths:
                if not os.path.exists(b_path):
                    os.mkdir(b_path)
                else: #start with an empty folder
                    shutil.rmtree(b_path)
                    os.makedirs(b_path) 

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

                if split == 'train':
                    path = args.train_data_path
                    n=args.n_train_samples
                elif split == 'val':
                    path = val_data_path 
                    n=args.n_val_samples  
                else:
                    raise NotImplementedError()
    
                ds = wds.DataPipeline(
                    wds.SimpleShardList(braceexpand(path)),
                    wds.tarfile_to_samples(),
                    wds.split_by_worker,
                    wds.split_by_node,
                ).with_length(n)

                dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
                num_batches =  ds.size // batch_size

                with tqdm.tqdm(total=num_batches) as pbar:
                    for (i,sample) in enumerate(dataloader):
                        #------------Features------------
                        rgb_features = [wds.torch_loads(item) for item in sample['rgb_features.pth']]
                        rgb_features = torch.stack(rgb_features , dim=0)
                        audio_features = [wds.torch_loads(item) for item in sample['audio_features.pth']]
                        audio_features = torch.stack(audio_features , dim=0)
                        data = {
                            "RGB": rgb_features,
                            "Audio": audio_features,
                        }
                        data = dict_to_cuda(data) #dict with RGB =(B, 3, 32, 224, 224), Audio=(B, 128, 128)

                        #------------Masks------------
                        rgb_mask = [wds.torch_loads(item) for item in sample['rgb_mask.pth']]
                        rgb_mask = torch.stack(rgb_mask , dim=0)
                        audio_mask = [wds.torch_loads(item) for item in sample['audio_mask.pth']]
                        audio_mask = torch.stack(audio_mask , dim=0)
                        masks = {
                            "RGB": rgb_mask,
                            "Audio": audio_mask,
                        }
                        masks = dict_to_cuda(masks) #dict with 'RGB'=(B, 512, 1), Audio=(B, 512, 1)
                        
                        #------------Pseudo Labels------------
                        keys = [s.split('__')[1] for s in sample['__key__']]
                        audio_pseudo, rgb_pseudo = get_pseudo_labels(keys, masks, args.deactivate_KL)
                        audio_pseudo = audio_pseudo.cuda() #(3086, )
                        rgb_pseudo = rgb_pseudo.cuda() #(3086, )

                        #------------Labels------------
                        labels = [int(item.decode()) for item in sample['action_label']] #(B, )
                        labels = torch.tensor(labels).cuda()

                        #------------Train Step------------
                        if split == "train":
                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                audio_pseudo,
                                rgb_pseudo,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                optim,
                                loss_fn,
                                kl_loss_fn,
                                scaler,
                                i,
                                num_batches,
                                args.gc,
                            )
                            #Save the prediction for each sample in the batch as pseudo_labels
                            if not args.deactivate_KL:
                                save_pseudo_labels(outputs, keys, masks)

                        #------------Validation Step------------
                        else:  #val
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
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
                    wandb.log({"{}/loss".format(split): total_loss / float(count), "{}/loss_epoch".format(split): epoch_i}) #epoch loss
                    wandb.log({"{}/acc".format(split): acc / float(count), "{}/acc_epoch".format(split): epoch_i}) #epoch accuracy 

            scheduler.step()
            wandb.log({"train/lr": scheduler.get_last_lr()[0]}) #epoch lr
            #wandb.log({"train/lr_epoch": epoch_i})

            if acc / float(count) > BestAcc or (args.save_all and epoch_i % 4 == 0): #save model every 4 epochs
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
    f.close()
