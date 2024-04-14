import torch
import numpy as np
import csv 
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Switching to CPU.")

csv_file_path1 = "epic-annotations/EPIC_100_train_full_half1.csv" #RGB samples
csv_file_path2 = "epic-annotations/EPIC_100_train_full_half2.csv" #Audio samples

sample_dict = {} #62429 samples vs 62297 samples
with open(csv_file_path1) as f: #read names of half1 samples with RGB modality: 31213 samples
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_dict[row[0]] = ["RGB"]
f.close()

with open(csv_file_path2) as f: #read names of half2 samples with Audio modality: 31084 samples
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_dict[row[0]] = ["Audio"]
f.close()


for sample in sample_dict.keys():
    if sample_dict[sample][0]=='RGB':
        save_path = "rgb_pseudo/{}.npy".format(sample) #P02_01_60.npy
    else:
        save_path = "audio_pseudo/{}.npy".format(sample)

    pseudo_label = np.load(save_path)
    if pseudo_label.shape[0]!=10:
        print(pseudo_label.shape)
        print(name)
