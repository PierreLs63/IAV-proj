from time import time
from datetime import datetime
start = time()
from mask_diffusion import generate_samples, create_monai_diffusion_unet, DiffusionProcess,prepare_monai_data_dicts, CacheDataset,get_monai_transforms
import torch
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader as MonaiDataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timesteps = 1000
    batch_size = 4


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
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = create_monai_diffusion_unet(
            img_channels=1,
            mask_channels=1,
            spatial_dims=2
        ).to(device)

    checkpoint = torch.load('weights/2mask_diffusion_model_final.pth') 
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion_process = DiffusionProcess(
            timesteps=timesteps,
            beta_schedule='scaled_linear_beta',  # MONAI schedule
            device=device
        )

    test_batch = next(iter(dataloader))
    test_masks = test_batch['mask'][:batch_size].to(device)


    images = generate_samples(model,diffusion_process,test_masks,batch_size)

    for i in range(batch_size):
        plt.imsave(f'samples/2_attention_layer/im{i}.png',images[i, 0].cpu().numpy(),cmap='gray')

if __name__ == '__main__':
    try : 
        main()
    except Exception as e:
        print( f"got error : {e}")    
    
    duration = time() - start
    with open("time_use.log",'a') as f:
        f.write(f"{datetime.now()}|{duration}\n")
        print("durée enregistrée")
