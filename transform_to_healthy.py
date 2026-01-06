from time import time
from datetime import datetime
start = time()
from mask_diffusion import generate_samples, create_monai_diffusion_unet, DiffusionProcess,prepare_monai_data_dicts, CacheDataset,get_monai_transforms
import torch
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from tqdm import tqdm
import numpy as np


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :",device)

    timesteps = 1000
    timestep_for_transform = 50
    BATCH_SIZE = 16
    n_att_layers = (False, False, True)

    print(f"generating {BATCH_SIZE} images with model of  attention layers")

    data_dicts = prepare_monai_data_dicts('cropped_centered')
        
    # Transformations (sans augmentation)
    train_transforms = get_monai_transforms()
    
    # CacheDataset pour accélérer le chargement
    dataset = CacheDataset(
        data=data_dicts,
        transform=train_transforms,
        cache_rate=0.5,  # Cache 50% des données en RAM
        num_workers=4
    )
        
    # DataLoader MONAI
    dataloader = MonaiDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = create_monai_diffusion_unet(
            img_channels=1,
            mask_channels=1,
            spatial_dims=2,
            attention_levels=n_att_layers
        ).to(device)

    checkpoint = torch.load(f'weights/mask_diffusion_cfg/diffusion_model_final.pth',  map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion_process = DiffusionProcess(
            timesteps=timesteps,
            beta_schedule='scaled_linear_beta',  # MONAI schedule
            device=device
        )

    test_batch = next(iter(dataloader))
    test_masks = test_batch['mask'][:BATCH_SIZE].to(device)
    test_images = test_batch['image'][:BATCH_SIZE].to(device)
    with torch.no_grad():

        t = torch.full((BATCH_SIZE,),timestep_for_transform, device=device,dtype = torch.int)
        shape = (BATCH_SIZE, 1, test_masks.shape[2], test_masks.shape[3])
        
        noisy_images = diffusion_process.add_noise(test_images,t)
        healthy_masks = torch.full_like(test_masks, 1,dtype = torch.int)
        
        iterator = torch.arange(timestep_for_transform - 1,-1,-1)
        for t in tqdm(iterator,desc= "denoising"):
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.float32)
                    # ===== CFG =====
            def model_forward(mask_value):
                inp = torch.cat([noisy_images, mask_value], dim=1)
                return model(inp, timesteps)

                
            eps_cond = model_forward(healthy_masks)
            eps_uncond = model_forward(torch.zeros_like(healthy_masks))  # classe 0 = no condition

            guidance_scale = 5.0

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            noisy_images, _ = diffusion_process.scheduler.step(eps, t, noisy_images)

        
    for i in range(BATCH_SIZE):
        fig, axes = plt.subplots(1, 4, figsize=(6,2))
        axes[0].imshow(test_images[i, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title(f'Original')
        axes[0].axis('off')

        axes[1].imshow(test_masks[i, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        axes[2].imshow(noisy_images[i, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title(f'Healthy version')
        axes[2].axis('off')

        orig_normalized = (test_images[i, 0] - test_images[i, 0].min()) / (test_images[i, 0].max() - test_images[i, 0].min()) * 255
        diff = np.abs(noisy_images.cpu().numpy().astype(float) - orig_normalized.cpu().numpy().astype(float))
        axes[3].imshow(diff[i, 0], cmap='hot')
        axes[3].set_title(f"Difference")
        axes[3].axis('off')

        plt.savefig(f'./healthy/im{i}.png', dpi=150)

if __name__ == '__main__':
    main()
