import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn.functional as torch_F

# MONAI imports
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandRotated, RandFlipd, RandZoomd, ToTensord, Lambda
)
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.utils import set_determinism


# ========== Dataset MONAI ==========
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


def get_monai_transforms(augmentation=True):
    """
    Créer les transformations MONAI pour les images médicales
    
    Args:
        augmentation: Si True, ajoute des augmentations de données
    
    Returns:
        Transformation MONAI composée
    """
    # Lambda pour charger les fichiers .npy
    load_npy = Lambda(func=lambda x: np.load(x).astype(np.float32))
    
    transforms_list = [
        # Charger les données .npy
        Lambda(func=lambda data: {
            'image': np.load(data['image']).astype(np.float32),
            'mask': np.load(data['mask']).astype(np.float32)
        }),
        # Assurer que les channels sont en premier [C, H, W]
        EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
        # Normaliser entre -1 et 1
        ScaleIntensityRanged(
            keys=['image', 'mask'],
            a_min=None,  # Calculer automatiquement
            a_max=None,
            b_min=-1.0,
            b_max=1.0,
            clip=True
        ),
    ]
    
    # Ajouter des augmentations si demandé
    if augmentation:
        transforms_list.extend([
            RandRotated(
                keys=['image', 'mask'],
                range_x=0.2,  # ±0.2 radians (~11 degrés)
                prob=0.5,
                mode=['bilinear', 'nearest'],
                padding_mode='zeros'
            ),
            RandFlipd(
                keys=['image', 'mask'],
                spatial_axis=0,  # Flip horizontal
                prob=0.5
            ),
            RandFlipd(
                keys=['image', 'mask'],
                spatial_axis=1,  # Flip vertical
                prob=0.5
            ),
            RandZoomd(
                keys=['image', 'mask'],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.5,
                mode=['area', 'nearest']
            ),
        ])
    
    return Compose(transforms_list)


# ========== Diffusion Utils ==========
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Schedule linéaire pour les betas"""
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Schedule cosinus pour les betas (plus stable)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionProcess:
    """Processus de diffusion forward et reverse avec MONAI scheduler"""
    
    def __init__(self, timesteps=1000, beta_schedule='scaled_linear_beta', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Utiliser le scheduler MONAI DDPM
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            schedule=beta_schedule,  # 'scaled_linear_beta' ou 'linear'
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Pour compatibilité avec l'ancien code
        self.betas = torch.tensor(self.scheduler.betas).to(device)
        self.alphas = torch.tensor(self.scheduler.alphas).to(device)
        self.alphas_cumprod = torch.tensor(self.scheduler.alphas_cumprod).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculs pour la diffusion forward
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculs pour la diffusion reverse
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: ajouter du bruit à l'image
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, mask, clip_denoised=True):
        """
        Reverse diffusion: un pas de débruitage
        Concatène le masque en entrée du modèle
        """
        batch_size = x_t.shape[0]
        
        # Concaténer x_t avec le masque pour conditionner la génération
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Prédire le bruit
        predicted_noise = model(model_input, t)
        
        # Extraire les coefficients
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # Calculer x_{t-1}
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, model, shape, mask, progress=True):
        """
        Générer une image complète en partant du bruit
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Partir du bruit pur
        img = torch.randn(shape, device=device)
        
        iterator = reversed(range(0, self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc='Sampling', total=self.timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, mask)
        
        return img
    
    def _extract(self, a, t, x_shape):
        """Extraire les valeurs de a correspondant aux indices t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ========== Model U-Net MONAI ==========
def create_monai_diffusion_unet(img_channels=1, mask_channels=1, spatial_dims=2):
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
        num_channels=(64, 128, 256, 512),  # Base channels à chaque niveau
        attention_levels=(False, False, True, True),  # Attention aux niveaux profonds
        num_res_blocks=2,  # Blocs résiduels par niveau
        num_head_channels=32,  # Pour l'attention
        with_conditioning=False,  # Pas de conditioning supplémentaire (on utilise le masque)
        resblock_updown=True,  # Utiliser des blocs résiduels pour up/down sampling
    )
    
    return model


# ========== FID Calculation ==========
class InceptionV3FeatureExtractor(nn.Module):
    """Extracteur de features Inception V3 pour le calcul du FID"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        # Charger Inception V3 pré-entraîné
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Utiliser la couche avant la classification (pool3)
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ).to(device)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x: [batch, 1, H, W] grayscale
        # Convertir en RGB et redimensionner à 299x299
        x = x.repeat(1, 3, 1, 1)  # Grayscale -> RGB
        x = torch_F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normaliser comme attendu par Inception
        x = (x + 1) / 2  # De [-1, 1] à [0, 1]
        x = (x - 0.5) / 0.5  # Normalisation Inception
        
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculer la distance de Fréchet entre deux gaussiennes multivariées"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Calculer sqrt(sigma1 @ sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Gérer les valeurs numériques instables
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Partie imaginaire due aux erreurs numériques
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_statistics(images, feature_extractor, batch_size=32):
    """Calculer moyenne et covariance des features"""
    feature_extractor.eval()
    
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            features = feature_extractor(batch)
            all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    
    return mu, sigma


def calculate_fid(real_images, generated_images, feature_extractor, batch_size=32, device='cuda'):
    """
    Calculer le FID entre images réelles et générées
    
    Args:
        real_images: Tensor [N, C, H, W] images réelles
        generated_images: Tensor [N, C, H, W] images générées
        feature_extractor: Modèle pour extraire les features
        batch_size: Taille des batchs pour le calcul
        device: Device pour les calculs
    
    Returns:
        fid_score: Score FID (plus bas = meilleur)
    """
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)
    
    # Calculer les statistiques pour les images réelles
    mu_real, sigma_real = compute_statistics(real_images, feature_extractor, batch_size)
    
    # Calculer les statistiques pour les images générées
    mu_gen, sigma_gen = compute_statistics(generated_images, feature_extractor, batch_size)
    
    # Calculer le FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_score


# ========== Training ==========
def train_diffusion(model, dataloader, diffusion_process, optimizer, device, num_epochs=100, 
                   fid_eval_freq=10, num_fid_samples=500):
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
    
    # Initialiser l'extracteur de features pour le FID
    print("Chargement du modèle Inception V3 pour le FID...")
    feature_extractor = InceptionV3FeatureExtractor(device=device)
    
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
    print(f"Images réelles préparées: {real_images_for_fid.shape}")
    
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
            
            # Forward diffusion: ajouter du bruit aux images
            x_noisy = diffusion_process.q_sample(images, t, noise)
            
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
        if (epoch + 1) % fid_eval_freq == 0:
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
                    
                    # Générer sans barre de progression pour chaque batch
                    generated_batch = diffusion_process.p_sample_loop(
                        model, shape, masks_batch, progress=False
                    )
                    generated_images_list.append(generated_batch)
                
                generated_images = torch.cat(generated_images_list, dim=0)
            
            # Calculer le FID
            fid_score = calculate_fid(
                real_images_for_fid,
                generated_images,
                feature_extractor,
                batch_size=32,
                device=device
            )
            
            fid_scores.append(fid_score)
            fid_epochs.append(epoch + 1)
            print(f'FID Score à l\'epoch {epoch+1}: {fid_score:.4f}\n')
            
            model.train()
        
        # Sauvegarder périodiquement
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'fid_scores': fid_scores,
                'fid_epochs': fid_epochs,
            }, f'diffusion_checkpoint_epoch_{epoch+1}.pth')
    
    return losses, fid_scores, fid_epochs


# ========== Sampling/Génération ==========
def generate_samples(model, diffusion_process, masks, num_samples=4, device='cuda'):
    """Générer des échantillons conditionnés par les masques"""
    
    model.eval()
    
    with torch.no_grad():
        # Shape: [batch, channels, H, W]
        shape = (num_samples, 1, masks.shape[2], masks.shape[3])
        
        # Générer les images
        generated = diffusion_process.p_sample_loop(model, shape, masks, progress=True)
    
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
    plt.savefig('diffusion_results.png', dpi=150)
    plt.show()


# ========== Main ==========
def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparamètres
    batch_size = 16
    num_epochs = 100
    learning_rate = 2e-4
    timesteps = 1000
    
    # Seed pour reproductibilité
    set_determinism(seed=42)
    
    # Dataset MONAI
    print('Préparation des données avec MONAI...')
    data_dicts = prepare_monai_data_dicts('cropped_centered')
    
    # Splits train/val (optionnel, ici on utilise tout pour l'entraînement)
    train_transforms = get_monai_transforms(augmentation=True)
    
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
        fid_eval_freq=10,  # Calculer le FID tous les 10 epochs
        num_fid_samples=500  # Utiliser 500 échantillons pour le FID
    )
    
    # Sauvegarder le modèle final
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fid_scores': fid_scores,
        'fid_epochs': fid_epochs,
    }, 'diffusion_model_final.pth')
    
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
    plt.savefig('training_metrics.png', dpi=150)
    plt.show()
    
    # Afficher les résultats FID
    if len(fid_scores) > 0:
        print('\n=== Scores FID ===')
        for epoch, score in zip(fid_epochs, fid_scores):
            print(f'Epoch {epoch}: FID = {score:.4f}')
        print(f'\nMeilleur FID: {best_fid:.4f} à l\'epoch {best_epoch}')
    
    print('\n=== Entraînement terminé ===')


if __name__ == '__main__':
    main()
