from time import time
from datetime import datetime
start = time()
from mask_diffusion import generate_samples, create_monai_diffusion_unet, DiffusionProcess,prepare_monai_data_dicts, CacheDataset,get_monai_transforms
import torch
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from tqdm import tqdm
import numpy as np

def build_cfg_condition(mask, drop_prob=0):
    """
    mask: [B, 1, H, W]
    return: [B, 1, H, W] values in {0,1,2}
    """

    infarct = (mask >= 3).int()
    no_inf  = (mask <= 2).int()


    # classes ∈ {1,2}
    classes = no_inf + infarct * 2   # [B,1,H,W]

    # Drop par batch
    drop = (torch.rand(mask.shape[0], 1, 1, 1, device=mask.device) < drop_prob).float()

    # Broadcast → OK
    classes = classes * (1.0 - drop)

    return classes.float()

   

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :",device)

    BATCH_SIZE = 32
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

    test_batch = next(iter(dataloader))
    test_masks = test_batch['mask'][:BATCH_SIZE].to(device)
    test_images = test_batch['image'][:BATCH_SIZE].to(device)
    cond_classes = build_cfg_condition(test_masks)
    
    labels = []
    for i in range(BATCH_SIZE):
        if (cond_classes[i] == 0).any() :
            labels.append("droped")
        elif (cond_classes[i] == 2).any():
            labels.append("unhealthy")
        else :
            labels.append("healthy")
    
            
    for i in range(BATCH_SIZE):
        fig, axes = plt.subplots(1, 3, figsize=(6,2))
        axes[0].imshow(test_images[i, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title(f'Original')
        axes[0].axis('off')

        axes[1].imshow(test_masks[i, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        axes[2].imshow(cond_classes[i, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title(f'CFG mask {labels[i]}')
        axes[2].axis('off')

       
        plt.savefig(f'plots/vis_mask/im{i}.png', dpi=150)

if __name__ == '__main__':
    main()
