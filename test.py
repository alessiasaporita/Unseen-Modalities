import torch
from dataloader_test import EPICKitchensTest
from ast_model import ASTModel
import pdb
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import datetime
from ast_configs import get_audio_configs
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
from vit import ViT
from feature_reorganization import ViTReorganization
from train import spatialtemporal2tokens

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_position",
        type=int,
        help="number of latent tokens",
        default=512,
    )
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
        "--save_name", type=str, help="name to save the predictions", default="1e4",
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_length = 128
    train_audio_configs, val_audio_configs = get_audio_configs(
        target_length=target_length
    )

    audio_model = ASTModel(
        label_dim=3806,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=target_length,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
    )
    audio_model = audio_model.to(device)
    audio_model = nn.DataParallel(audio_model)
    #checkpoint = torch.load("checkpoints/audio.pt")
    #audio_model.load_state_dict(checkpoint["model"])
    audio_model.eval()

    rgb_model = omnivore_swinT(pretrained=True) #changed
    rgb_model.heads = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
    )
    rgb_model.multimodal_model = False
    rgb_model = torch.nn.DataParallel(rgb_model)
    #checkpoint = torch.load("checkpoints/rgb.pt")
    #rgb_model.load_state_dict(checkpoint['state_dict'])
    rgb_model = rgb_model.to(device)
    rgb_model.eval()

    multimodal_model = ViT(num_classes = 3806, dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device)

    reorganization_module = ViTReorganization(dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device)

    checkpoint = torch.load("checkpoints/best_multimodal1e4.pt")
    multimodal_model.load_state_dict(checkpoint['model'])
    multimodal_model.eval()

    reorganization_module.load_state_dict(checkpoint['reorganization'])
    reorganization_module.eval()

    test_dataset = EPICKitchensTest(audio_conf=val_audio_configs, split="test", audio_data_path=args.audio_data_path, rgb_data_path=args.rgb_data_path, num_position = args.num_position)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    num_of_samples = len(test_dataloader)
    acc = 0
    save_path = 'predictions/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    pred_path = "predictions/{}.csv".format(args.save_name)
    num1 = 5

    with open(pred_path, "a") as f:
        for i, (rgb_data, audio, action_label, rgb_mask, audio_mask) in enumerate(test_dataloader):
            audio = audio.cuda().squeeze(0) #(B, num_of_fbanks, 128, 128)->(num_of_fbanks, 128, 128)
            rgb_data = rgb_data.cuda().squeeze(0) #(B, num_of_fbanks, 3, 32, 224, 224)->(num_of_fbanks, 3, 32, 224, 224)
            rgb_mask = rgb_mask.cuda().squeeze(0) #(B, num_of_fbanks, 512, 1)->(num_of_fbanks, 512, 1)
            audio_mask = audio_mask.cuda().squeeze(0) #(B, num_of_fbanks, 512, 1)->(num_of_fbanks, 512, 1)

            output_predictions = []
            with torch.no_grad(): #consider multimodal prediction for 5 fbanks and rgb at a time
                for k in range(audio.size()[0] // num1 + 1): #num_of_fbanks // 5 + 1 
                    if k*num1 >= audio.size()[0]: 
                        break
                    audio_outputs = audio_model(audio[k*num1:(k+1)*num1]) # [num_of_fbanks, 146, 768] --> audio[k*num1:(k+1)*num1] = audio[0:5]->audio[5:10]...
                    
                    rgb_outputs = rgb_model(rgb_data[k*num1:(k+1)*num1]) #[num_of_fbanks, 784, 16, 7, 7]
                    rgb_outputs = spatialtemporal2tokens(rgb_outputs) #(num_of_fbanks, 784, 768)

                    rgb_outputs, audio_outputs = reorganization_module(rgb_outputs, audio_outputs) #rgb = (num_of_fbanks, 512, 768), audio =(num_of_fbanks, 512, 768)
                    #audio_mask[:,:,:] = 0.0
                    outputs = multimodal_model(rgb_outputs, audio_outputs, rgb_mask, audio_mask) #(num_of_fbanks, 2, 3086)
                    
                    outputs = torch.softmax(outputs, dim=-1) #(num_of_fbanks, 2, 3086)
                    output_predictions.append(outputs.detach())

            predictions = torch.cat(output_predictions, dim=0) #(num_of_fbanks * (num_of_fbanks//5+1), 2, 3086)
            predictions = torch.mean(predictions, dim=0) #(2, 3086)
            predictions = predictions.detach().cpu().numpy()
            action_label = action_label.numpy()[0]

            #index of the maximum value in the flattened array.
            #This means it will return the index of the maximum value considering all elements of the array as if they were in a single, one-dimensional array.
            if np.argmax(predictions) == action_label:
                acc += 1
            print(i+1, '/', num_of_samples, 'Accuracy:', acc / (i+1))
            
            f.write(
                "{}/{}, Accuracy:, {}\n".format(
                    i+1,
                    num_of_samples,
                    acc / (i+1),
                )
                    )
            f.flush()
    f.close()
    
            