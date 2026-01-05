import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Charger les résultats
df = pd.read_csv('mse_results_1.csv')

# Trier par MSE pour voir les meilleures et pires correspondances
df_sorted = df.sort_values('mse')

# Afficher les 10 meilleures correspondances (MSE la plus faible)
print("10 meilleures correspondances (MSE la plus faible):")
print(df_sorted.head(10)[['generated_image', 'original_case', 'original_image', 'mse']])

# Sélectionner quelques paires à visualiser
n_pairs = 5
pairs_to_show = df_sorted.head(n_pairs)

# Créer la figure
fig, axes = plt.subplots(n_pairs, 3, figsize=(12, 3*n_pairs))
if n_pairs == 1:
    axes = axes.reshape(1, -1)

for idx, (_, row) in enumerate(pairs_to_show.iterrows()):
    # Charger l'image générée
    gen_path = Path('samples') / row['model'] / row['generated_image']
    gen_img = Image.open(gen_path).convert('L')
    
    # Charger l'image originale
    orig_path = Path('cropped_centered') / row['original_case'] / 'Slice' / 'Image' / row['original_image']
    orig_img = np.load(orig_path)
    
    # Afficher l'image générée
    axes[idx, 0].imshow(gen_img, cmap='gray')
    axes[idx, 0].set_title(f"Générée\n{row['generated_image']}")
    axes[idx, 0].axis('off')
    
    # Afficher l'image originale
    axes[idx, 1].imshow(orig_img, cmap='gray')
    axes[idx, 1].set_title(f"Originale\n{row['original_case']}\n{row['original_image']}")
    axes[idx, 1].axis('off')
    
    # Afficher la différence absolue
    gen_array = np.array(gen_img)
    # Normaliser l'image originale pour la comparaison visuelle
    orig_normalized = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255
    diff = np.abs(gen_array.astype(float) - orig_normalized.astype(float))
    axes[idx, 2].imshow(diff, cmap='hot')
    axes[idx, 2].set_title(f"Différence\nMSE: {row['mse']:.4f}")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('meilleures_correspondances.png', dpi=150, bbox_inches='tight')
print("\n✓ Image sauvegardée: meilleures_correspondances.png")
plt.show()

# Afficher aussi les 5 pires correspondances
print("\n\n5 pires correspondances (MSE la plus élevée):")
pairs_worst = df_sorted.tail(5)
print(pairs_worst[['generated_image', 'original_case', 'original_image', 'mse']])

fig2, axes2 = plt.subplots(5, 3, figsize=(12, 15))

for idx, (_, row) in enumerate(pairs_worst.iterrows()):
    # Charger l'image générée
    gen_path = Path('samples') / row['model'] / row['generated_image']
    gen_img = Image.open(gen_path).convert('L')
    
    # Charger l'image originale
    orig_path = Path('cropped_centered') / row['original_case'] / 'Slice' / 'Image' / row['original_image']
    orig_img = np.load(orig_path)
    
    # Afficher l'image générée
    axes2[idx, 0].imshow(gen_img, cmap='gray')
    axes2[idx, 0].set_title(f"Générée\n{row['generated_image']}")
    axes2[idx, 0].axis('off')
    
    # Afficher l'image originale
    axes2[idx, 1].imshow(orig_img, cmap='gray')
    axes2[idx, 1].set_title(f"Originale\n{row['original_case']}\n{row['original_image']}")
    axes2[idx, 1].axis('off')
    
    # Afficher la différence absolue
    gen_array = np.array(gen_img)
    orig_normalized = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255
    diff = np.abs(gen_array.astype(float) - orig_normalized.astype(float))
    axes2[idx, 2].imshow(diff, cmap='hot')
    axes2[idx, 2].set_title(f"Différence\nMSE: {row['mse']:.4f}")
    axes2[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('pires_correspondances.png', dpi=150, bbox_inches='tight')
print("\n✓ Image sauvegardée: pires_correspondances.png")
plt.show()
