import torch
from unimodals.dataloader_test import EPICKitchensTest
from unimodals.ast_model import ASTModel
import pdb
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
import warnings
import torch.nn.functional as F
import datetime
from ast_configs import get_audio_configs
from unimodals.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT


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
        "--modality", type=str, help="audio or rgb", default="rgb",
    ) 
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
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
    
    if args.modality=="audio":
        model = ASTModel(
            label_dim=3806,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=target_length,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="base384",
        )
    else:
        model = omnivore_swinT(pretrained=False) 
        model.heads = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
        )
        model.multimodal_model = False

    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

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

    with open(pred_path, "a") as f:
        for i, (rgb_data, audio, action_label) in enumerate(test_dataloader):
            with torch.no_grad(): 
                if args.modality=='audio':
                    audio = audio.cuda()
                    outputs = model(audio)
                else:
                    rgb_data = rgb_data.cuda()
                    outputs = model(rgb_data) 
            outputs = torch.softmax(outputs, dim=-1) #(B, 3086)
            predictions = outputs.detach().cpu().numpy()
            action_label = action_label.numpy()[0]

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
    
            