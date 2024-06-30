from ast_model import ASTModel
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
import webdataset as wds
from dataloader_train import EPICKitchensTrain
from dataloader_validation import EPICKitchensValidation
from ast_configs import get_audio_configs
from tqdm import tqdm
import random
import json
import argparse
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import random
import numpy as np
import csv
import os
import torch.nn as nn
from ast_configs import get_audio_configs

def spatialtemporal2tokens(data):
    b, c, f, h, w = data.size()
    data = data.view(b, c, f * h * w)
    data = data.transpose(1, 2).contiguous()
    return data

def extract_features(unimodal_models, data):
    outputs = {}
    for key, value in data.items(): #key = Audio, RGB
        outputs[key] = unimodal_models[key](value)
        if key == 'RGB':
            outputs[key] = spatialtemporal2tokens(outputs[key])
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='standard')
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--job_idx', type=int, default=0)
    parser.add_argument('--job_size', type=int, default=5000)
    parser.add_argument(
        "--resume_rgb_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
    )
    parser.add_argument(
        "--resume_audio_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
    )

    args = parser.parse_args()
    print(args)

    shard_start = args.job_idx * args.job_size
    shard_end = shard_start + args.job_size
    split = args.split

    pattern = '/work/tesi_asaporita/webdataset/epic_kitchens-' + split + '-%03d.tar' % args.job_idx
    num_position = 512
    target_length=128
    train_audio_configs, val_audio_configs = get_audio_configs(
        target_length=target_length
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = EPICKitchensTrain(
        audio_conf=train_audio_configs,
        split="train",
        audio_data_path = "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted",
        rgb_data_path = "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS",
        num_position=num_position,
    )
    val_dataset = EPICKitchensValidation(
        audio_conf=val_audio_configs,
        split="validation",
        audio_data_path = "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted",
        rgb_data_path = "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS",
        num_position=num_position,
    )

    if split == 'training':
        dataset = train_dataset
        split_i = 0
    elif split == 'validation':
        dataset = val_dataset
        split_i = 1
    else:
        split_i = 2

    indexes = list(range(len(dataset)))
    if split_i == 0 or split_i == 1: #shuffle train and validation
        random.Random(4).shuffle(indexes)

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
    #audio_model = audio_model.to(device)
    audio_model = nn.DataParallel(audio_model)
    checkpoint = torch.load(args.resume_audio_checkpoint, map_location=torch.device('cpu'))
    audio_model.load_state_dict(checkpoint["model"])
    audio_model.eval()

    #rgb
    rgb_model = omnivore_swinT(pretrained=True) 
    rgb_model.heads = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
    )
    rgb_model.multimodal_model = False
    rgb_model = torch.nn.DataParallel(rgb_model)
    checkpoint = torch.load(args.resume_rgb_checkpoint, map_location=torch.device('cpu'))
    rgb_model.load_state_dict(checkpoint['model'])
    #rgb_model = rgb_model.to(device)
    rgb_model.eval()

    unimodal_models = { #audio, RGB
        'RGB': rgb_model,
        'Audio': audio_model
    }
    
    dst = wds.TarWriter(pattern) 
    inserted = -1 
    for idx in tqdm(indexes):
        inserted += 1 
        if inserted >= shard_end:
            print("end my shard")
            break
        elif inserted < shard_end and inserted >= shard_start:
            data, action_label, masks, audio_pseudo, rgb_pseudo, keys = dataset.get_value_by_index(idx)
            
            data['Audio']=data['Audio'].unsqueeze(0)
            data['RGB']=data['RGB'].unsqueeze(0)
            with torch.no_grad():
                outputs = extract_features(unimodal_models, data)

            if split == 'training':
                sample = {
                    '__key__': str(idx)+'__'+keys,
                    'rgb_features.pth': outputs['RGB'].squeeze(0), #tensor (784, 768) of float32
                    'rgb_mask.pth': torch.tensor(masks['RGB']),
                    'rgb_pseudo.pth': rgb_pseudo,
                    'audio_features.pth': outputs['Audio'].squeeze(0), #tensor (146, 768) of float 32
                    'audio_mask.pth': torch.tensor(masks['Audio']),
                    'audio_pseudo.pth': audio_pseudo,
                    'action_label.id': action_label,
                }
            else: #validation
                sample = {
                    '__key__': str(idx)+'__'+keys,
                    'rgb_features.pth': outputs['RGB'].squeeze(0), #tensor (784, 768) of float32
                    'rgb_mask.pth': torch.tensor(masks['RGB']),
                    'audio_features.pth': outputs['Audio'].squeeze(0), #tensor (146, 768) of float 32
                    'audio_mask.pth': torch.tensor(masks['Audio']),
                    'action_label.id': action_label,
                }
            dst.write(sample)
    dst.close()




