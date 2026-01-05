from time import time
from datetime import datetime
start = time()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# MONAI imports
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Lambda
)
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.utils import set_determinism, first
from monai.metrics import compute_frechet_distance
from monai.inferers import DiffusionInferer

FILENAME = __file__.split("/")[-1].split(".")[0]
PATH_PLOTS = Path("./plots/" + FILENAME)
PATH_WEIGHTS = Path("./weights/" + FILENAME)
os.makedirs(os.path.dirname(PATH_PLOTS), exist_ok=True)
os.makedirs(os.path.dirname(PATH_WEIGHTS), exist_ok=True)

# ========== Dataset MONAI ==========
def load_and_normalize(data):
    """Charger et normaliser les fichiers .npy"""
    image = np.load(data['image']).astype(np.float32)
    mask = np.load(data['mask']).astype(np.float32)
    
    # Ajouter dimension channel [H, W] -> [1, H, W]
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    
    # Normaliser entre -1 et 1
    image_min, image_max = image.min(), image.max()
    if image_max > image_min:
        image = 2.0 * (image - image_min) / (image_max - image_min) - 1.0
    
    mask_min, mask_max = mask.min(), mask.max()
    if mask_max > mask_min:
        mask = 2.0 * (mask - mask_min) / (mask_max - mask_min) - 1.0
    
    return {'image': image, 'mask': mask}


def prepare_monai_data_dicts(root_dir):
    """
    Préparer les dictionnaires de données pour MONAI
    
    Args:
        root_dir: chemin vers cropped_centered/
    
    Returns:
        Liste de dictionnaires avec clés 'image' et 'mask'
    """
    root_dir = Path(root_dir)
    data_dicts = []
    
    # Parcourir tous les dossiers Case_*
    for case_dir in sorted(root_dir.glob("Case_*")):
        image_dir = case_dir / "Slice" / "Image"
        mask_dir = case_dir / "Slice" / "Mask"
        
        if image_dir.exists() and mask_dir.exists():
            # Récupérer tous les fichiers npy
            image_files = sorted(image_dir.glob("slice_*.npy"))
            
            for img_file in image_files:
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    data_dicts.append({
                        'image': str(img_file),
                        'mask': str(mask_file)
                    })
    
    print(f"Dataset préparé : {len(data_dicts)} slices trouvées")
    return data_dicts


def get_monai_transforms():
    """
    Créer les transformations MONAI pour les images médicales
    
    Returns:
        Transformation MONAI composée
    """
    transforms_list = [
        Lambda(func=load_and_normalize)
    ]
    
    return Compose(transforms_list)

# ========== Diffusion Process avec MONAI ==========
class DiffusionProcess:
    """Processus de diffusion utilisant MONAI scheduler et inferer"""
    
    def __init__(self, timesteps=1000, beta_schedule='linear_beta', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        
        # Utiliser le scheduler MONAI DDPM
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            schedule=beta_schedule,
            beta_start=0.0015,
           beta_end=0.0195
        )
        
        # Utiliser l'inferer MONAI pour la génération
        self.inferer = DiffusionInferer(self.scheduler)
    
    def add_noise(self, x_start, t, noise=None):
        """Ajouter du bruit avec le scheduler MONAI"""
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.scheduler.add_noise(original_samples=x_start, noise=noise, timesteps=t)
    
    def sample(self, model, shape, mask, progress=True):
        """Générer des images avec l'inferer MONAI"""
        device = next(model.parameters()).device
        
        # Fonction wrapper pour passer le masque au modèle
        def model_with_mask(x, t):
            model_input = torch.cat([x, mask], dim=1)
            return model(model_input, timesteps=t)
        
        # Générer avec l'inferer MONAI
        noise = torch.randn(shape, device=device)
        
        if progress:
            iterator = tqdm(self.scheduler.timesteps, desc='Sampling')
        else:
            iterator = self.scheduler.timesteps
        
        sample = noise
        for t in iterator:
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            model_output = model_with_mask(sample, timesteps)
            sample, _ = self.scheduler.step(model_output, t, sample)
        
        return sample


# ========== Model U-Net MONAI ==========
def create_monai_diffusion_unet(img_channels=1, mask_channels=1, spatial_dims=2, attention_levels=(False, False, True)):
    """
    Créer un DiffusionModelUNet de MONAI configuré pour notre tâche
    
    Args:
        img_channels: Nombre de channels de l'image (1 pour grayscale)
        mask_channels: Nombre de channels du masque (1 pour grayscale)
        spatial_dims: 2D ou 3D (2 pour nos slices)
    
    Returns:
        DiffusionModelUNet configuré
    """
    # Input = img_channels + mask_channels (concaténation)
    in_channels = img_channels + mask_channels
    out_channels = img_channels  # On prédit le bruit pour l'image seulement
    
    model = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(128, 256, 512),
        attention_levels=attention_levels,
        num_res_blocks=2,
        num_head_channels=(0, 256, 512),
    )
    
    return model


# ========== FID Calculation avec MONAI ==========
def prepare_images_for_fid(images):
    """
    Préparer les images pour le calcul du FID
    Convertir grayscale en RGB et redimensionner à 299x299
    """
    # Convertir de [-1, 1] à [0, 1]
    images = (images + 1) / 2
    
    # Convertir grayscale en RGB si nécessaire
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    
    # Redimensionner à 299x299 (taille attendue par Inception)
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    return images


def extract_inception_features(images, device='cuda'):
    """
    Extraire les features Inception V3 pour le calcul du FID
    
    Args:
        images: Tensor d'images au format [B, 3, 299, 299] dans [0, 1]
    
    Returns:
        Features extraites
    """
    from torchvision.models import inception_v3
    
    # Charger Inception V3
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # Retirer la couche finale
    inception = inception.to(device)
    inception.eval()
    
    with torch.no_grad():
        features = inception(images)
    
    return features


def compute_fid_statistics(features):
    """
    Calculer la moyenne et la covariance des features
    
    Args:
        features: Tensor de features [N, D]
    
    Returns:
        mu, sigma
    """
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.T)
    return mu, sigma


# ========== Training ==========
def train_diffusion(model, dataloader, diffusion_process, optimizer, device, num_epochs=100, 
                   fid_eval_freq=10, num_fid_samples=100):
    """
    Entraîner le modèle de diffusion avec évaluation FID
    
    Args:
        model: Modèle de diffusion
        dataloader: DataLoader pour les données
        diffusion_process: Processus de diffusion
        optimizer: Optimiseur
        device: Device (cuda/cpu)
        num_epochs: Nombre d'epochs
        fid_eval_freq: Fréquence d'évaluation du FID (tous les N epochs)
        num_fid_samples: Nombre d'échantillons pour le calcul du FID
    """
    
    model.train()
    losses = []
    fid_scores = []
    fid_epochs = []
    
    # Préparer un ensemble fixe d'images réelles pour le FID
    print(f"Préparation de {num_fid_samples} images réelles pour l'évaluation FID...")
    real_images_list = []
    real_masks_list = []
    for batch_data in dataloader:
        # MONAI retourne des dictionnaires
        real_images_list.append(batch_data['image'])
        real_masks_list.append(batch_data['mask'])
        if len(real_images_list) * batch_data['image'].shape[0] >= num_fid_samples:
            break
    
    real_images_for_fid = torch.cat(real_images_list, dim=0)[:num_fid_samples].to(device)
    real_masks_for_fid = torch.cat(real_masks_list, dim=0)[:num_fid_samples].to(device)
    
    # Préparer les images réelles pour le FID (format Inception)
    real_images_fid_format = prepare_images_for_fid(real_images_for_fid)
    print(f"Images réelles préparées: {real_images_for_fid.shape} -> {real_images_fid_format.shape}")
    
    # Extraire les features des images réelles une seule fois
    print("Extraction des features Inception pour les images réelles...")
    real_features = extract_inception_features(real_images_fid_format, device)
    real_mu, real_sigma = compute_fid_statistics(real_features)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_data in progress_bar:
            # MONAI retourne un dictionnaire
            images = batch_data['image'].to(device)
            masks = batch_data['mask'].to(device)
            
            batch_size = images.shape[0]
            
            # Échantillonner des timesteps aléatoires
            t = torch.randint(0, diffusion_process.timesteps, (batch_size,), device=device)
            
            # Générer du bruit
            noise = torch.randn_like(images)
            
            # Forward diffusion: ajouter du bruit aux images avec MONAI
            x_noisy = diffusion_process.add_noise(images, t, noise)
            
            # Concaténer avec le masque pour conditionner
            model_input = torch.cat([x_noisy, masks], dim=1)
            
            # Prédire le bruit (MONAI attend timesteps normalisés)
            predicted_noise = model(model_input, timesteps=t)
            
            # Loss MSE entre le bruit prédit et le vrai bruit
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
        
        # Calculer le FID périodiquement
        if (epoch + 1) % fid_eval_freq == 0 or epoch + 1 == 10 :
            print(f"\nCalcul du FID à l'epoch {epoch+1}...")
            model.eval()
            
            # Générer des images
            with torch.no_grad():
                generated_images_list = []
                batch_size_gen = 16
                num_batches = (num_fid_samples + batch_size_gen - 1) // batch_size_gen
                
                for i in tqdm(range(num_batches), desc='Génération pour FID'):
                    start_idx = i * batch_size_gen
                    end_idx = min(start_idx + batch_size_gen, num_fid_samples)
                    current_batch_size = end_idx - start_idx
                    
                    masks_batch = real_masks_for_fid[start_idx:end_idx]
                    shape = (current_batch_size, 1, masks_batch.shape[2], masks_batch.shape[3])
                    
                    # Générer avec MONAI inferer
                    generated_batch = diffusion_process.sample(
                        model, shape, masks_batch, progress=False
                    )
                    generated_images_list.append(generated_batch)
                
                generated_images = torch.cat(generated_images_list, dim=0)
            
            # Préparer les images générées pour le FID
            generated_images_fid_format = prepare_images_for_fid(generated_images)
            
            # Extraire les features des images générées
            generated_features = extract_inception_features(generated_images_fid_format, device)
            gen_mu, gen_sigma = compute_fid_statistics(generated_features)
            
            # Calculer le FID avec compute_frechet_distance de MONAI
            fid_score = compute_frechet_distance(gen_mu, gen_sigma, real_mu, real_sigma).item()
            
            fid_scores.append(fid_score)
            fid_epochs.append(epoch + 1)
            print(f'FID Score à l\'epoch {epoch+1}: {fid_score:.4f}\n')
            
            model.train()
        

    return losses, fid_scores, fid_epochs


# ========== Sampling/Génération ==========
def generate_samples(model, diffusion_process, masks, num_samples=4, device='cuda'):
    """Générer des échantillons conditionnés par les masques"""
    
    model.eval()
    
    with torch.no_grad():
        # Shape: [batch, channels, H, W]
        shape = (num_samples, 1, masks.shape[2], masks.shape[3])
        
        # Générer les images avec MONAI
        generated = diffusion_process.sample(model, shape, masks, progress=True)
    
    return generated


def visualize_results(images, masks, generated, num_samples=4):
    """Visualiser les résultats"""
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Image originale
        axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Masque
        axes[1, i].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
        
        # Image générée
        axes[2, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title(f'Generated {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(PATH_PLOTS / 'diffusion_results.png', dpi=150)


# ========== Main ==========
def main():
    start = time()
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparamètres
    batch_size = 16
    num_epochs = 6000
    learning_rate = 2e-4
    timesteps = 1000
    
    # Seed pour reproductibilité
    set_determinism(seed=42)
    
    # Dataset MONAI
    print('Préparation des données avec MONAI...')
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

    # Modèle MONAI
    print('Création du modèle MONAI DiffusionModelUNet...')
    model = create_monai_diffusion_unet(
        img_channels=1,
        mask_channels=1,
        spatial_dims=2
    ).to(device)

    try :
        checkpoint = torch.load( PATH_WEIGHTS / 'diffusion_model_final.pth') 
        model.load_state_dict(checkpoint['model_state_dict'])
    except :
        pass
    
    print(f'Nombre de paramètres: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # Processus de diffusion avec scheduler MONAI
    diffusion_process = DiffusionProcess(
        timesteps=timesteps,
        beta_schedule='scaled_linear_beta',  # MONAI schedule
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Entraînement
    print('\n=== Début de l\'entraînement ===\n')
    losses, fid_scores, fid_epochs = train_diffusion(
        model=model,
        dataloader=dataloader,
        diffusion_process=diffusion_process,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        fid_eval_freq=1000,  # Calculer le FID tous les 10 epochs
        num_fid_samples=100  # Utiliser 100 échantillons pour le FID
    )
    
    # Sauvegarder le modèle final
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fid_scores': fid_scores,
        'fid_epochs': fid_epochs,
    },  PATH_WEIGHTS / 'diffusion_model_final.pth')
    
    # Génération d'échantillons
    print('\n=== Génération d\'échantillons ===\n')
    
    # Prendre quelques masques du dataset (MONAI retourne des dicts)
    test_batch = next(iter(dataloader))
    test_images = test_batch['image'][:4].to(device)
    test_masks = test_batch['mask'][:4].to(device)
    
    # Générer des images conditionnées par ces masques
    generated = generate_samples(
        model=model,
        diffusion_process=diffusion_process,
        masks=test_masks,
        num_samples=4,
        device=device
    )
    
    # Visualiser
    visualize_results(test_images, test_masks, generated)
    
    # Plot de la loss et du FID
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # FID
    if len(fid_scores) > 0:
        ax2.plot(fid_epochs, fid_scores, marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FID Score')
        ax2.set_title('FID Score (lower is better)')
        ax2.grid(True)
        
        # Afficher le meilleur score
        best_fid = min(fid_scores)
        best_epoch = fid_epochs[fid_scores.index(best_fid)]
        ax2.axhline(y=best_fid, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best: {best_fid:.2f} @ epoch {best_epoch}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PATH_PLOTS / 'training_metrics.png', dpi=150)
    
    # Afficher les résultats FID
    if len(fid_scores) > 0:
        print('\n=== Scores FID ===')
        for epoch, score in zip(fid_epochs, fid_scores):
            print(f'Epoch {epoch}: FID = {score:.4f}')
        print(f'\nMeilleur FID: {best_fid:.4f} à l\'epoch {best_epoch}')
 
    print('\n=== Entraînement terminé ===')


if __name__ == '__main__':
    try : 
        main()
    except Exception as e:
        print( f"got error : {e}")    
    
    duration = time() - start
    with open("time_use.log",'a') as f:
        f.write(f"{datetime.now()}|{duration}\n")
        print("durée enregistrée")
