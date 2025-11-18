import os
import shutil
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torchinfo import summary
from monai import transforms
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized
from monai.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet, AutoencoderKL
from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from monai.networks.schedulers import DDPMScheduler
from monai.utils import first
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
# Parameters
spatial_dims = 2
in_channels = 1
out_channels = 1
channels = (128, 128, 256)
latent_channels = 3
num_res_blocks = 2
norm_num_groups = channels[0]
attention_levels = (False, False, False)
with_encoder_nonlocal_attn = False
with_decoder_nonlocal_attn = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use of the with command line to avoid the automatic display of log information during the instanciation of the class model 
with contextlib.redirect_stdout(None):
    autoencoderkl = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        attention_levels=attention_levels,
        with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
        with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
    )
    autoencoderkl = autoencoderkl.to(device, dtype=torch.float32)
    print("AE device:", next(autoencoderkl.parameters()).device)
    # MONAI nécessite un déplacement manuel de certains sous-modules
if hasattr(autoencoderkl, "encoder"):
    autoencoderkl.encoder.to(device)
if hasattr(autoencoderkl, "decoder"):
    autoencoderkl.decoder.to(device)
if hasattr(autoencoderkl, "quant_conv"):
    autoencoderkl.quant_conv.to(device)
if hasattr(autoencoderkl, "post_quant_conv"):
    autoencoderkl.post_quant_conv.to(device)
    
# Print the summary of the encoder network
summary_kwargs = dict(
    col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0
)
summary(autoencoderkl, (1, 1, image_size, image_size), device=str(device), **summary_kwargs)

unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=2,
    channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
).to(device)
latent_to_unet = nn.Conv2d(latent_channels, 1, kernel_size=1).to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)


check_data = first(train_loader)
with torch.no_grad():
    images = check_data["image"].to(device, dtype=torch.float32)
    z_mu, z_sigma = autoencoderkl.encode(images)   # z_mu.shape = (B, latent_channels, H, W)
    z = autoencoderkl.sampling(z_mu, z_sigma) 

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)
inferer = LatentDiffusionInferer(
    scheduler=scheduler,
    scale_factor=scale_factor
)
optimizer = torch.optim.Adam(list(unet.parameters()) + list(latent_to_unet.parameters()), lr=1e-4)

max_epochs = 200
val_interval = 40
epoch_losses = []
val_losses = []
scaler = GradScaler()

for epoch in range(max_epochs):
    unet.train()
    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma)
            z = F.interpolate(z, size=(16, 16), mode="bilinear", align_corners=False)
            noise = torch.randn_like(z)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (z.shape[0],), device=device).long()
        
        # <- ici on passe le modèle et l'autoencoder à l'appel
            noise_pred = inferer(
            inputs=z,
            diffusion_model=unet,
            noise=noise,
            timesteps=timesteps,
            autoencoder_model=autoencoderkl
        )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))
    # Validation
    if (epoch + 1) % val_interval == 0:
         unet.eval()
         val_loss = 0
         with torch.no_grad():
             for _val_step, batch in enumerate(val_loader, start=1):
                 images = batch["image"].to(device).float()
                 z_mu, z_sigma = autoencoderkl.encode(images)
                 z = autoencoderkl.sampling(z_mu, z_sigma)
                 z_unet = latent_to_unet(z)
                 z_unet = F.interpolate(z_unet, size=(16, 16), mode="bilinear", align_corners=False)
                 noise = torch.randn_like(z_unet)
                 timesteps = torch.randint(
                         0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                 noise_pred = inferer(
                         inputs=z_unet,
                         diffusion_model=unet,
                         noise=noise,
                         timesteps=timesteps,
                         autoencoder_model=autoencoderkl,
                     )
                 loss = F.mse_loss(noise_pred.float(), noise.float())

                 val_loss += loss.item()
         val_loss /= _val_step
         val_losses.append(val_loss)
         print(f"Epoch {epoch} val loss: {val_loss:.4f}")

         # Sampling image during training
         z_sample = torch.randn((1, latent_channels, 16, 16), device=device)
         z_sample_unet = latent_to_unet(z_sample)
         scheduler.set_timesteps(num_inference_steps=1000)
         with autocast("cuda", enabled=True):
            decoded = inferer.sample(
                input_noise=z_sample_unet,
                diffusion_model=unet,
                scheduler=scheduler,
                autoencoder_model=autoencoderkl
            )

         plt.figure(figsize=(2, 2))
         plt.style.use("default")
         plt.imshow(decoded[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
         plt.tight_layout()
         plt.axis("off")
         plt.show()
progress_bar.close()
