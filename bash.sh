#!/bin/bash
#python train.py --lr 1e-1 --batch_size 96 --save_name 1e-1
#python train.py --lr 1e-2 --batch_size 96 --save_name 1e-2resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-1.pt
#python train.py --lr 1e-3 --batch_size 96 --save_name 1e-3resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-2resumed.pt

python unimodal_train.py --lr 1e-5 --modality audio --save_all True --batch_size 8 --gc 6 --num_epochs 50 --save_name _audio_ --workers 16 --audio_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 
python unimodal_train.py --lr 1e-4 --modality rgb --save_all True --batch_size 2 --gc 48 --num_epochs 120 --save_name _rgb --workers 16 --audio_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 

python train.py --lr 1e-4 --save_all True --batch_size 8 --gc 12 --num_epochs 120 --save_name _KL_ --audio_data_path "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted" --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 
python train.py --lr 1e-4 --deactivate_KL True --save_all True --batch_size 8 --gc 12 --num_epochs 120 --save_name _no_KL_ --audio_data_path "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted" --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 
python train_base_web.py --lr 1e-4 --save_all True --batch_size 8 --gc 12 --num_epochs 120 --save_name _base_ --workers 16 --audio_data_path "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted" --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 