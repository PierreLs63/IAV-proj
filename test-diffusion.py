import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ========== Dataset ==========
class SliceDataset(Dataset):
    """Dataset pour charger les slices d'images et leurs masques"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: chemin vers cropped_centered/
            transform: transformations optionnelles
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Parcourir tous les dossiers Case_*
        for case_dir in sorted(self.root_dir.glob("Case_*")):
            image_dir = case_dir / "Slice" / "Image"
            mask_dir = case_dir / "Slice" / "Mask"
            
            if image_dir.exists() and mask_dir.exists():
                # Récupérer tous les fichiers npy
                image_files = sorted(image_dir.glob("slice_*.npy"))
                
                for img_file in image_files:
                    mask_file = mask_dir / img_file.name
                    if mask_file.exists():
                        self.samples.append((img_file, mask_file))
        
        print(f"Dataset chargé : {len(self.samples)} slices trouvées")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Charger l'image et le masque
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        
        # Normaliser entre -1 et 1
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
            image = image * 2 - 1
        
        if mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask * 2 - 1
        
        # Ajouter dimension channel si nécessaire
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        
        # Convertir en tenseurs
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask


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
    """Processus de diffusion forward et reverse"""
    
    def __init__(self, timesteps=1000, beta_schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Définir le schedule des betas
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = self.betas.to(device)
        
        # Calculer les alphas et leurs produits cumulés
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
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


# ========== Model U-Net ==========
class TimeEmbedding(nn.Module):
    """Embedding temporel pour le timestep"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timesteps):
        # Créer un embedding sinusoïdal
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """Bloc résiduel avec embedding temporel"""
    
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        
        # Ajouter l'embedding temporel
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class UNetDiffusion(nn.Module):
    """
    U-Net pour la diffusion avec concaténation du masque
    Input: [image_bruitée, masque] concaténés sur la dimension des channels
    """
    
    def __init__(self, img_channels=1, mask_channels=1, base_channels=64, 
                 time_dim=256, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        
        self.time_dim = time_dim
        self.time_mlp = TimeEmbedding(time_dim)
        
        # Input a img_channels + mask_channels (concaténation)
        input_channels = img_channels + mask_channels
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_channels = []
        
        channels = [input_channels] + [base_channels * m for m in channel_mults]
        
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            self.encoder.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, time_dim),
                ResidualBlock(out_ch, out_ch, time_dim),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # Downsample
            ]))
            self.encoder_channels.append(out_ch)
        
        # Bottleneck
        bottleneck_ch = channels[-1]
        self.bottleneck = nn.ModuleList([
            ResidualBlock(bottleneck_ch, bottleneck_ch, time_dim),
            ResidualBlock(bottleneck_ch, bottleneck_ch, time_dim)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        
        for i in range(len(channels) - 2, -1, -1):
            out_ch = channels[i]
            in_ch = channels[i + 1]
            
            self.decoder.append(nn.ModuleList([
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),  # Upsample
                ResidualBlock(out_ch * 2, out_ch, time_dim),  # *2 pour skip connection
                ResidualBlock(out_ch, out_ch, time_dim)
            ]))
        
        # Output layer
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], img_channels, 1)  # Output = nb de channels de l'image
        )
    
    def forward(self, x, timesteps):
        """
        x: [batch, img_channels + mask_channels, H, W] (image + masque concaténés)
        timesteps: [batch]
        """
        # Embedding temporel
        time_emb = self.time_mlp(timesteps)
        
        # Encoder avec skip connections
        skips = []
        h = x
        
        for res1, res2, downsample in self.encoder:
            h = res1(h, time_emb)
            h = res2(h, time_emb)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        for res in self.bottleneck:
            h = res(h, time_emb)
        
        # Decoder avec skip connections
        for (upsample, res1, res2), skip in zip(self.decoder, reversed(skips)):
            h = upsample(h)
            h = torch.cat([h, skip], dim=1)
            h = res1(h, time_emb)
            h = res2(h, time_emb)
        
        return self.output(h)


# ========== Training ==========
def train_diffusion(model, dataloader, diffusion_process, optimizer, device, num_epochs=100):
    """Entraîner le modèle de diffusion"""
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            batch_size = images.shape[0]
            
            # Échantillonner des timesteps aléatoires
            t = torch.randint(0, diffusion_process.timesteps, (batch_size,), device=device)
            
            # Générer du bruit
            noise = torch.randn_like(images)
            
            # Forward diffusion: ajouter du bruit aux images
            x_noisy = diffusion_process.q_sample(images, t, noise)
            
            # Concaténer avec le masque pour conditionner
            model_input = torch.cat([x_noisy, masks], dim=1)
            
            # Prédire le bruit
            predicted_noise = model(model_input, t)
            
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
        
        # Sauvegarder périodiquement
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'diffusion_checkpoint_epoch_{epoch+1}.pth')
    
    return losses


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
    
    # Dataset
    dataset = SliceDataset(
        root_dir='cropped_centered',
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Modèle
    model = UNetDiffusion(
        img_channels=1,
        mask_channels=1,
        base_channels=64,
        time_dim=256,
        channel_mults=(1, 2, 4, 8)
    ).to(device)
    
    print(f'Nombre de paramètres: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # Processus de diffusion
    diffusion_process = DiffusionProcess(
        timesteps=timesteps,
        beta_schedule='cosine',
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Entraînement
    print('\n=== Début de l\'entraînement ===\n')
    losses = train_diffusion(
        model=model,
        dataloader=dataloader,
        diffusion_process=diffusion_process,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )
    
    # Sauvegarder le modèle final
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'diffusion_model_final.pth')
    
    # Génération d'échantillons
    print('\n=== Génération d\'échantillons ===\n')
    
    # Prendre quelques masques du dataset
    test_images, test_masks = next(iter(dataloader))
    test_images = test_images[:4].to(device)
    test_masks = test_masks[:4].to(device)
    
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
    
    # Plot de la loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print('\n=== Entraînement terminé ===')


if __name__ == '__main__':
    main()
