from time import time
from datetime import datetime
start = time()
from mask_diffusion import generate_samples, create_monai_diffusion_unet, DiffusionProcess,prepare_monai_data_dicts, CacheDataset,get_monai_transforms
import torch
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader as MonaiDataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :",device)

    timesteps = 1000
    BATCH_SIZE = 100
    NUM_LAYER = "1"
    n_att_layers = (False, True, True) if NUM_LAYER == "2" else (False, False, True)

    print(f"generating {BATCH_SIZE} images with model of {NUM_LAYER} attention layers")

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

    checkpoint = torch.load(f'weights/mask_diffusion/diffusion_model_final.pth') 
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion_process = DiffusionProcess(
            timesteps=timesteps,
            beta_schedule='scaled_linear_beta',  # MONAI schedule
            device=device
        )

    test_batch = next(iter(dataloader))
    test_masks = test_batch['mask'][:BATCH_SIZE].to(device)

    print("generating ...")
    images = generate_samples(model,diffusion_process,test_masks,BATCH_SIZE)

    for i in range(BATCH_SIZE):
        plt.imsave(f'samples/simple/im{i}.png',images[i, 0].cpu().numpy(),cmap='gray')

if __name__ == '__main__':
    try : 
        main()
    except Exception as e:
        print( f"got error : {e}")    
    
    duration = time() - start
    with open("time_use.log",'a') as f:
        f.write(f"{datetime.now()}|{duration}\n")
        print("durée enregistrée")
