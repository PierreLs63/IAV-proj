import os
import shutil
import contextlib
import numpy as np
import torch
import torch.nn as nn
from monai import transforms
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Préparer dossiers train/val slice-wise
# ---------------------------
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

# ---------------------------
# 2️⃣ Construire les listes de slices
# ---------------------------
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

print(f"Total train slices: {len(train_datalist)}")
print(f"Total val slices: {len(val_datalist)}")


image_size = 64  # adapte si besoin !

train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityd(keys=["image"]),
    transforms.RandAffined(
        keys=["image"],
        rotate_range=[(-np.pi/36, np.pi/36), (-np.pi/36, np.pi/36)],
        translate_range=[(-5, 5), (-5, 5)],
        scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
        spatial_size=[image_size, image_size],
        padding_mode="zeros",
        prob=0.5,
    ),
])

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityd(keys=["image"]),
])


batch_size = 32

train_dataset = Dataset(data=train_datalist, transform=train_transforms)
val_dataset   = Dataset(data=val_datalist, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, persistent_workers=False)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, persistent_workers=False)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


batch = next(iter(train_loader))
images = batch["image"].numpy()  # shape: (B,1,H,W)
print("Batch shape:", images.shape)

fig = plt.figure(figsize=(20, 4))
for idx in range(min(20, len(images))):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap="gray")
plt.show()