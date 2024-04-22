#!/bin/bash
python train.py --lr 1e-1 --batch_size 96 --save_name 1e-1
python train.py --lr 1e-2 --batch_size 96 --save_name 1e-2resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-1.pt
python train.py --lr 1e-3 --batch_size 96 --save_name 1e-3resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-2resumed.pt


python train.py --lr 1e-4 --resume_training True --save_all True --resume_checkpoint checkpoints/best_multimodal_KL_resumed2.pt --batch_size 32 --gc 3 --num_epochs 120 --save_name _KL_resumed3 --audio_data_path "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted" --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS 
python train.py --lr 1e-4 --resume_training True --save_all True --resume_checkpoint checkpoints/best_multimodal_no_KL_resumed1.pt --batch_size 32 --gc 3 --num_epochs 120 --save_name _no_KL_resumed2 --audio_data_path "/work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS-Audio-Extracted" --rgb_data_path /work/pnrr_fair/umi/epic_kitchens/EPIC-KITCHENS --deactivate_KL True 