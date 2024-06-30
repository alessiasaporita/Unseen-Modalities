from unimodals.ast_model import ASTModel
from unimodals.dataloader_train import EPICKitchensTrain
from unimodals.dataloader_validation import EPICKitchensValidation
import torch
import argparse
import tqdm
import os
import shutil
import numpy as np
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from ast_configs import get_audio_configs
from unimodals.omnivore_model import omnivore_swinT
import wandb
import os
from torch.optim.lr_scheduler import MultiStepLR


def save_pseudo_labels(outputs, keys, modality):
    detached_outputs = outputs.detach().cpu()
    detached_outputs = torch.softmax(detached_outputs, dim=-1) 

    #For each sample in the batch, save its relative prediction
    for i in range(len(keys)): 
        #------------RGB------------
        if modality=='rgb': #RGB 
            save_path = "/work/tesi_asaporita/checkpoint/EK/rgb/rgb_pseudo/{}.npy".format(keys[i])
            if os.path.exists(save_path): #predictions for i-th sample 
                rgb_pseudo = np.load(save_path)
                if rgb_pseudo.shape[0]>=40:
                    rgb_pseudo=rgb_pseudo[-39:]
                rgb_pseudo = np.concatenate((rgb_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                rgb_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("/work/tesi_asaporita/checkpoint/EK/rgb/rgb_pseudo/{}.npy".format(keys[i]), rgb_pseudo)
        #------------Audio------------
        else: #Audio 
            save_path = "/work/tesi_asaporita/checkpoint/EK/audio/audio_pseudo/{}.npy".format(keys[i])
            if os.path.exists(save_path):
                audio_pseudo = np.load(save_path)
                if audio_pseudo.shape[0]>=40:
                    audio_pseudo=audio_pseudo[-39:]
                audio_pseudo = np.concatenate((audio_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                audio_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("/work/tesi_asaporita/checkpoint/EK/audio/audio_pseudo/{}.npy".format(keys[i]), audio_pseudo)

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing #0.1
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def train_one_step(
    data,
    labels,
    model,
    optim,
    loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
):
    outputs = model(data) 

    #Total loss
    output_loss = loss = loss_fn(outputs, labels) 
    
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
    model,
    loss_fn,
):
    with torch.no_grad():
        outputs = model(data)
        output_loss = loss = loss_fn(outputs, labels) 
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
        "--modality", type=str, help="audio or rgb", default="rgb",
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
        name='Unimodal',
        config={
        "modality":args.modality,
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

    base_path = "/work/tesi_asaporita/checkpoint/EK/"
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
    if args.modality=="audio":
    #audio
        model = ASTModel( 
            label_dim=3806,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=target_length, #128
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="base384",
        ) 
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        milestones = list(range(5, args.num_epochs))
        scheduler = MultiStepLR(optim, milestones=milestones, gamma=0.85)
    else:
    #rgb
        model = omnivore_swinT(pretrained=True) 
        model.heads = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
        ) 
        model.multimodal_model = False
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optim, milestones=[70], gamma=0.1)

    model = model.to(device)
    model = nn.DataParallel(model)
    
    loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    scaler = GradScaler()
    BestLoss = float("inf")
    initial_epoch = 0
    BestEpoch = 0
    BestAcc = 0

    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) ######
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_epoch = checkpoint['epoch'] + 1
        BestLoss = checkpoint['best_loss']
        BestAcc = checkpoint['best_acc']
        
    train_loader = torch.utils.data.DataLoader(
        EPICKitchensTrain(
            audio_conf=train_audio_configs,
            modality=args.modality, ######
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
                model.train(split == "train")

                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i,(data, labels, keys),) in enumerate(dataloaders[split]):
                        #data = dict with RGB =(B, 3, 32, 224, 224), Audio=(B, 128, 128)
                        labels = labels.cuda() #(B, )

                        if args.modality=="rgb":
                            data = data['RGB'].cuda()
                        else:
                            data = data['Audio'].cuda()
                        
                        if split == "train":
                            output, loss = train_one_step(
                                data = data,
                                labels=labels,
                                model=model,
                                optim=optim,
                                loss_fn=loss_fn,
                                scaler=scaler,
                                indice=i,
                                last_indice=len(dataloaders[split]),
                                gc=args.gc,
                            )
                        else:
                            output, loss = val_one_step(
                                data=data,
                                labels=labels,
                                model=model,
                                loss_fn=loss_fn,
                            )

                        wandb.log({"{}/step_loss".format(split): loss}) #step loss
                        total_loss += loss.item() * batch_size

                        if split == "train": 
                            #Save the prediction for each sample in the batch as pseudo_labels
                            save_pseudo_labels(output, keys, args.modality)

                        outputs = torch.softmax(output, dim=-1) #(B, 3086)
                        _, predict = torch.max(outputs, dim=1)
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += outputs.size()[0]

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
            #wandb.log({"train/lr_epoch": epoch_i})

            if acc / float(count) > BestAcc: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestAcc = acc / float(count)
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
