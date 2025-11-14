import os
import shutil
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized, ScaleIntensityd
from sklearn.model_selection import train_test_split
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

root_input = "cropped_centered"      
root_output = "data/MedDataset"

train_dir = os.path.join(root_output, "training")
val_dir = os.path.join(root_output, "validation")
if os.path.exists(root_output):
    shutil.rmtree(root_output)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

patients = [d for d in os.listdir(root_input) if os.path.isdir(os.path.join(root_input, d))]
patients.sort()

# Séparation 80/20
train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=42)

def copy_slices(src_root, dst_root, patients_list):
    for p in patients_list:
        src_img = os.path.join(src_root, p, "Slice", "Image")
        dst_img = os.path.join(dst_root, p, "Slice", "Image")
        if os.path.exists(src_img):
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copytree(src_img, dst_img)
        else:
            print(f"⚠️ {src_img} manquant, ignoré.")

copy_slices(root_input, train_dir, train_patients)
copy_slices(root_input, val_dir, val_patients)

print(f"✅ {len(train_patients)} patients pour train, {len(val_patients)} pour validation")


def build_datalist_from_slices(data_root):
    datalist = []
    for patient in os.listdir(data_root):
        img_folder = os.path.join(data_root, patient, "Slice", "Image")
        if os.path.exists(img_folder):
            for f in sorted(os.listdir(img_folder)):
                datalist.append({"image": os.path.join(img_folder, f), "label": 0})
    return datalist

train_datalist = build_datalist_from_slices(train_dir)
val_datalist = build_datalist_from_slices(val_dir)

print(f"Nombre de slices training: {len(train_datalist)}")
print(f"Nombre de slices validation: {len(val_datalist)}")


all_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),  # met chaque slice entre 0 et 1
    Resized(keys=["image"], spatial_size=(64,64))
])

train_dataset = Dataset(data=train_datalist, transform=all_transforms)
val_dataset = Dataset(data=val_datalist, transform=all_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

import matplotlib.pyplot as plt

batch = next(iter(train_loader))
images = batch["image"].numpy()  # shape [B, C, H, W]
print(f"Batch size: {images.shape}")

fig, axes = plt.subplots(1, min(8, images.shape[0]), figsize=(15,3))
for i in range(min(8, images.shape[0])):
    axes[i].imshow(images[i,0,:,:], cmap="gray")
    axes[i].axis("off")
plt.savefig("./plots/imgs.png")
