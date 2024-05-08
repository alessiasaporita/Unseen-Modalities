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

available_list = []
with open("epic-annotations/available_sound.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        available_list.append(row[0])
f.close()

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

samples = []
with open("epic-annotations/EPIC_100_train.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        if row[2] not in available_list and row[0] in sample_dict and "Audio" in sample_dict[row[0]]:
            continue
        if row[0] not in sample_dict:
            continue
        samples.append(row)
f.close()
with open("epic-annotations/EPIC_100_validation.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        if row[2] not in available_list and row[0] in sample_dict and "Audio" in sample_dict[row[0]]:
            continue
        if row[0] not in sample_dict:
            continue
        samples.append(row)
f.close()

for i,sample in enumerate(samples):
    available_modalities = sample_dict[sample[0]]
    if "RGB" in available_modalities:
        save_path = "/work/tesi_asaporita/UnseenModalities/rgb_pseudo/{}.npy".format(sample[0]) #P02_01_60.npy
        pseudo_label = np.load(save_path)
        print("{}:{}".format(sample[0], pseudo_label.shape))
    else:
        save_path = "/work/tesi_asaporita/UnseenModalities/audio_pseudo/{}.npy".format(sample[0])

    #pseudo_label = np.load(save_path)
    #print("{}:{}".format(sample[0], pseudo_label.shape))
