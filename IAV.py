import os
import shutil
import contextlib
import numpy as np
import torch
import torch.nn as nn
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

# ---------------------------
# 3️⃣ Transforms MONAI pour slices 2D
# ---------------------------
image_size = 64
all_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),  # normalisation 0-1 automatique
    Resized(keys=["image"], spatial_size=(image_size,image_size))
])

train_dataset = Dataset(data=train_datalist, transform=all_transforms)
val_dataset = Dataset(data=val_datalist, transform=all_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# ---------------------------
# 4️⃣ Model setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder
spatial_dims = 2
in_channels = 1
out_channels = 1
channels = (16, 32, 64)
latent_channels = 3
num_res_blocks = 2
norm_num_groups = channels[0]
attention_levels = (False, False, False)

with contextlib.redirect_stdout(None):
    model = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        attention_levels=attention_levels,
    ).to(device)

# Discriminateur
num_layers_d = 3
channels_d = 16

with contextlib.redirect_stdout(None):
    discriminator = PatchDiscriminator(
        spatial_dims=spatial_dims,
        num_layers_d=num_layers_d,
        channels=channels_d,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=(Act.LEAKYRELU, {"negative_slope":0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    ).to(device)

# ---------------------------
# 5️⃣ Losses
# ---------------------------
p_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
adv_loss = PatchAdversarialLoss(criterion="least_squares").to(device)
l1_loss = nn.L1Loss().to(device)

def vae_gaussian_kl_loss(mu, sigma):
    kl_loss = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1,2,3])
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return kl_loss

def reconstruction_loss(x_reconstructed, x):
    return l1_loss(x_reconstructed.float(), x.float())

def perceptual_loss(x_reconstructed, x):
    return p_loss(x_reconstructed.float(), x.float())

def loss_function(recon_x, x, mu, sigma, kl_weight, p_weight, a_weight, logits, target_is_real, for_discriminator):
    recon = reconstruction_loss(recon_x, x)
    kld = vae_gaussian_kl_loss(mu, sigma)
    pl = perceptual_loss(recon_x, x)
    adv = adv_loss(logits, target_is_real=target_is_real, for_discriminator=for_discriminator)
    return recon + kl_weight*kld + p_weight*pl + a_weight*adv, recon.item(), kl_weight*kld.item(), p_weight*pl.item(), a_weight*adv.item()

# ---------------------------
# 6️⃣ Optimizers
# ---------------------------
learning_rate_g = 1e-4
learning_rate_d = 5e-4

optimizer_generator = torch.optim.Adam(model.parameters(), lr=learning_rate_g)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)

# ---------------------------
# 7️⃣ Training parameters
# ---------------------------
n_epochs = 120
kl_weight = 1e-6
perceptual_weight = 1e-3
adversarial_weight = 1e-2

train_generator_loss_list = []
train_discriminator_loss_list = []
valid_metric_list = []

best_valid_metric = float('inf')
best_model = None
best_epoch = 0

# ---------------------------
# 8️⃣ Training loop
# ---------------------------
if __name__ == "__main__":
    for epoch in range(n_epochs):
        model.train()
        discriminator.train()

        train_generator_loss = 0
        train_discriminator_loss = 0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)

            # ---- Generator ----
            optimizer_generator.zero_grad()
            reconstruction, z_mu, z_sigma = model(inputs)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

            loss_generator, recon_val, kld_val, perceptual_val, adversarial_val = loss_function(
                reconstruction, 
                inputs, 
                z_mu, 
                z_sigma, 
                kl_weight, 
                perceptual_weight,
                adversarial_weight,
                logits_fake,
                target_is_real=True,
                for_discriminator=False)

            loss_generator.backward()
            optimizer_generator.step()
            train_generator_loss += loss_generator.item() * inputs.size(0)

            # ---- Discriminator ----
            if adversarial_weight > 0:
                optimizer_discriminator.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                logits_real = discriminator(inputs.contiguous().detach())[-1]

                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_discriminator = adversarial_weight * discriminator_loss

                loss_discriminator.backward()
                optimizer_discriminator.step()
                train_discriminator_loss += loss_discriminator.item() * inputs.size(0)

        train_generator_loss_list.append(train_generator_loss / len(train_loader.dataset))
        if adversarial_weight > 0:
            train_discriminator_loss_list.append(train_discriminator_loss / len(train_loader.dataset))

        # ---- Validation ----
        model.eval()
        valid_metric = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                reconstruction, _, _ = model(inputs)
                recon_val = reconstruction_loss(reconstruction, inputs)
                valid_metric += recon_val.item() * inputs.size(0)

        valid_metric_list.append(valid_metric / len(val_loader.dataset))

        print(f"Epoch [{epoch+1}/{n_epochs}] "
              f"Train G Loss: {train_generator_loss_list[-1]:.6f} "
              f"Train D Loss: {(train_discriminator_loss_list[-1] if adversarial_weight>0 else 0):.6f} "
              f"Valid Loss: {valid_metric_list[-1]:.6f}")

        # Save best model
        if valid_metric_list[-1] < best_valid_metric:
            best_valid_metric = valid_metric_list[-1]
            best_model = model.state_dict()
            best_epoch = epoch + 1

    # Load and save best model
    model.load_state_dict(best_model)
    torch.save(best_model, "best_hand_vae_model.pth")
    print(f"✅ Best model saved from epoch {best_epoch} with validation loss {best_valid_metric:.6f}")
