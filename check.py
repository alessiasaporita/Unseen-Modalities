from ast_model import ASTModel
from dataloader_train import EPICKitchensTrain
from dataloader_validation import EPICKitchensValidation
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
import warnings
import torch.nn.functional as F
from ast_configs import get_audio_configs
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
import sys
import csv
from braceexpand import braceexpand
import webdataset as wds

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

def extract_features(unimodal_models, data):
    outputs = {}
    for key, value in data.items(): #key = Audio, RGB
        outputs[key] = unimodal_models[key](value)
        if key == 'RGB':
            outputs[key] = spatialtemporal2tokens(outputs[key])
    return outputs

def equal_check(tensor1, tensor2):
    if torch.equal(tensor1, tensor2):
        return True
    else:
        return False


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument(
        "--audio_data_path",
        type=str,
        help="path to data",
        default="/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted",
    )
    parser.add_argument(
        "--rgb_data_path",
        type=str,
        help="path to data",
        default="/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS",
    )
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
        "--num_position", type=int, help="number of projection tokens", default=512,
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
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
    audio_model.eval()

    #rgb
    rgb_model = omnivore_swinT(pretrained=True) 
    rgb_model.heads = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
    )
    rgb_model.multimodal_model = False
    rgb_model.eval()

    unimodal_models = { #audio, RGB
        'RGB': rgb_model,
        'Audio': audio_model
    }

    train_dataset = EPICKitchensTrain(
            audio_conf=train_audio_configs,
            split="train",
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
            deactivate_KL=True,
        )
    val_dataset = EPICKitchensValidation(
            audio_conf=val_audio_configs,
            split="validation",
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
        )

    datasets = {"train": train_dataset, "val": val_dataset}
    
    print("---------------Start Training---------------")
    for split in ["train"]:
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
        dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=4)#, pin_memory=True)
        num_batches =  ds.size // batch_size
        print("fino a qui tutto bene!!!!!!")
        with tqdm.tqdm(total=num_batches, file=sys.stdout) as pbar:
            for (i,sample) in enumerate(dataloader):
                print("Iteration:", i)
                keys = [s.split('__')[1] for s in sample['__key__']]
                indexes = [int(s.split('__')[0]) for s in sample['__key__']]
                #------------Features------------
                rgb_features = [wds.torch_loads(item) for item in sample['rgb_features.pth']]
                rgb_features = torch.stack(rgb_features , dim=0)
                audio_features = [wds.torch_loads(item) for item in sample['audio_features.pth']]
                audio_features = torch.stack(audio_features , dim=0)
                
                #------------Masks------------
                rgb_mask = [wds.torch_loads(item) for item in sample['rgb_mask.pth']]
                rgb_mask = torch.stack(rgb_mask , dim=0)
                audio_mask = [wds.torch_loads(item) for item in sample['audio_mask.pth']]
                audio_mask = torch.stack(audio_mask , dim=0)

                #------------Labels------------
                labels = [int(item.decode()) for item in sample['action_label']] #(B, )
                labels = torch.tensor(labels)
                index = indexes[0]

                data, action_label, masks, _ = datasets[split].get_value_by_index(index)

                if labels[0]!=action_label:
                    print("Labels not equal for {}".format(keys[0]))
                    #raise Exception()
                else:
                    print("Label OK")

                if not equal_check(rgb_mask.squeeze(0), torch.tensor(masks["RGB"])):
                    print("RGB masks not equal for {}".format(keys[0]))
                    #raise Exception()
                else:
                    print("RGB mask OK")
                
                if not equal_check(audio_mask.squeeze(0), torch.tensor(masks["Audio"])):
                    print("Audio masks not equal for {}".format(keys[0]))
                    #raise Exception()
                else:
                    print("Audio mask OK")

                if split=='val':
                    data['RGB']=data['RGB'].unsqueeze(0) 
                   
                    with torch.no_grad():
                        rgb_output = rgb_model(data['RGB'])
                        rgb_output = spatialtemporal2tokens(rgb_output) 
                       
                    if abs(rgb_output.sum()-rgb_features.sum())>10:
                        print("RGB features not equal for {}".format(keys[0]))
                        #raise Exception()
                    else:
                        print("OK: difference of {}".format(abs(rgb_output.sum()-rgb_features.sum())))
                    
                pbar.update()
        
                    
