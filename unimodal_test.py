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
        "--modality", type=str, help="audio or rgb", default="rgb",
    ) 
    parser.add_argument(
        "--save_name", type=str, help="name to save the predictions", default="1e4",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, help="path to the checkpoint file", default="checkpoints/best_multimodal_KL",
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
        model = nn.DataParallel(model)
        checkpoint = torch.load("checkpoints/audio.pt")
    else:
        model = omnivore_swinT(pretrained=False) #changed
        model.heads = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
        )
        model.multimodal_model = False
        model = torch.nn.DataParallel(rgb_model)
        checkpoint = torch.load("checkpoints/rgb.pt")
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
    num1 = 5

    with open(pred_path, "a") as f:
        for i, (rgb_data, audio, action_label) in enumerate(test_dataloader):
            audio = audio.squeeze(0) #(B, num_of_fbanks, 128, 128)->(num_of_fbanks, 128, 128)
            rgb_data = rgb_data.squeeze(0) #(B, num_of_fbanks, 3, 32, 224, 224)->(num_of_fbanks, 3, 32, 224, 224)

            output_predictions = []
            with torch.no_grad(): #consider multimodal prediction for 5 fbanks and rgb at a time
                for k in range(audio.size()[0] // num1 + 1): #num_of_fbanks // 5 + 1 
                    if k*num1 >= audio.size()[0]: 
                        break
                    if args.modality=='audio':
                        audio = audio.cuda()
                        outputs = model(audio[k*num1:(k+1)*num1]) # [num_of_fbanks, 146, 768] --> audio[k*num1:(k+1)*num1] = audio[0:5]->audio[5:10]...
                    else:
                        rgb_data = rgb_data.cuda()
                        outputs = model(rgb_data[k*num1:(k+1)*num1]) #[num_of_fbanks, 784, 16, 7, 7]
            
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
    
            