import torch
import argparse
import tqdm
import os, io, sys
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
import math
import os
from torch.optim.lr_scheduler import MultiStepLR

def dict_to_cuda(data):
    for key, value in data.items():
        data[key] = value.cuda()
    return data


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


def train_one_step(
    data,
    labels,
    masks,
    multimodal_model,
    reorganization_module,
    optim,
    loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
):
    rgb, audio = reorganization_module( #feature projection = (B, 512, 256) = (B, k*, d*)
        data['RGB'], data['Audio'] 
    ) 

    outputs = multimodal_model( #(B, 2, 3086), the two CLS tokens
        rgb, audio, masks['RGB'], masks['Audio']
    ) 

    output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 
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
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="path to train data",
        default="/work/tesi_asaporita/UnseenModalities/webdataset/epic_kitchens-training-{000..020}.tar",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="path to validation data",
        default="/work/tesi_asaporita/UnseenModalities/webdataset/epic_kitchens-validation-{000..004}.tar",
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
        name='Unseen Modalities WEB Base',
        config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gc": args.gc,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_training": args.resume_training,
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
    loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    optim = torch.optim.SGD(
        list(multimodal_model.parameters())+list(reorganization_module.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4
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
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) 
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_epoch = checkpoint['epoch'] + 1
        BestLoss = checkpoint['best_loss']
        BestAcc = checkpoint['best_acc']
    

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

                if split == 'train':
                    path = args.train_data_path
                    n=args.n_train_samples
                elif split == 'val':
                    path = args.val_data_path 
                    n=args.n_val_samples  
                else:
                    raise NotImplementedError()
    
                ds = wds.DataPipeline(
                    wds.SimpleShardList(braceexpand(path)),
                    wds.tarfile_to_samples(),
                    wds.split_by_worker,
                    wds.split_by_node,
                ).with_length(n)

                if split=='train':
                    dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "rgb_features.pth", "rgb_mask.pth", "rgb_pseudo.pth", "audio_features.pth", "audio_mask.pth", "audio_pseudo.pth", "action_label.id")
                else:
                    dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "rgb_features.pth", "rgb_mask.pth", "audio_features.pth", "audio_mask.pth", "action_label.id")
                num_batches =  math.ceil(ds.size/batch_size)

                with tqdm.tqdm(total=num_batches) as pbar:
                    for (i,sample) in enumerate(dataloader):
                        if split=='train':
                            keys, rgb_features, rgb_mask, rgb_pseudo, audio_features, audio_mask, audio_pseudo, action_label = sample
                        else:
                            keys, rgb_features, rgb_mask, audio_features, audio_mask, action_label = sample

                        #------------Features------------
                        rgb_features = [wds.torch_loads(item) for item in rgb_features]
                        rgb_features = torch.stack(rgb_features , dim=0)
                        audio_features = [wds.torch_loads(item) for item in audio_features]
                        audio_features = torch.stack(audio_features , dim=0)
                        data = {
                            "RGB": rgb_features,
                            "Audio": audio_features,
                        }
                        data = dict_to_cuda(data) #dict with RGB =(B, 784, 768), Audio=(B, 146, 768)

                        #------------Masks------------
                        rgb_mask = [wds.torch_loads(item) for item in rgb_mask] 
                        rgb_mask = torch.stack(rgb_mask , dim=0)
                        audio_mask = [wds.torch_loads(item) for item in audio_mask] 
                        audio_mask = torch.stack(audio_mask , dim=0)
                        masks = {
                            "RGB": rgb_mask,
                            "Audio": audio_mask,
                        }
                        masks = dict_to_cuda(masks) #dict with 'RGB'=(B, 512, 1), Audio=(B, 512, 1)

                        #------------Labels------------
                        labels = [int(item.decode()) for item in action_label] #(B, )
                        labels = torch.tensor(labels).cuda()

                        #------------Train Step------------
                        if split == "train":
                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                multimodal_model,
                                reorganization_module,
                                optim,
                                loss_fn,
                                scaler,
                                i,
                                num_batches,
                                args.gc,
                            )

                        #------------Validation Step------------
                        else:  #val
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
                                multimodal_model,
                                reorganization_module,
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

            if acc / float(count) > BestAcc: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestAcc = acc / float(count)
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
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
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
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
