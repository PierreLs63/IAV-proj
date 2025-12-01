from time import time
from datetime import datetime
start = time()
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm


def load_npy_image(npy_path):
    """Charge une image .npy et la convertit en tenseur PyTorch."""
    img = np.load(npy_path)
    # Normaliser entre 0 et 1 si nécessaire
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).float()


def load_png_image(png_path):
    """Charge une image PNG et la convertit en tenseur PyTorch."""
    img = Image.open(png_path).convert('L')  # Convertir en niveaux de gris
    img = np.array(img)
    # Normaliser entre 0 et 1
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).float()


def calculate_mse(img1, img2):
    """Calcule la MSE entre deux images en utilisant PyTorch."""
    # S'assurer que les images ont la même forme
    if img1.shape != img2.shape:
        print(f"Attention: les images n'ont pas la même forme: {img1.shape} vs {img2.shape}")
        # Redimensionner si nécessaire
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # Calculer la MSE avec PyTorch
    mse = torch.nn.functional.mse_loss(img1, img2)
    return mse.item()


def main():
    samples_path = Path("samples")
    cropped_path = Path("cropped_centered")
    
    # Vérifier l'existence des dossiers
    if not samples_path.exists():
        print(f"Erreur: Le dossier 'samples' n'existe pas.")
        return None
    
    if not cropped_path.exists():
        print(f"Erreur: Le dossier 'cropped_centered' n'existe pas.")
        return None
    
    sample_folders = []
    for folder in samples_path.iterdir():
        if folder.is_dir() and list(folder.glob("*.png")):
            sample_folders.append(folder)
    
    sample_folders = sorted(sample_folders)
    
    if not sample_folders:
        print(f"Aucun dossier avec des images PNG trouvé dans 'samples'")
        return None
    
    print(f"Dossiers de samples trouvés: {[f.name for f in sample_folders]}")
    
    results = []
    
    # Parcourir chaque dossier de samples trouvé
    for layer_folder in sample_folders:
        print(f"\nTraitement de {layer_folder.name}...")
        
        # Récupérer toutes les images PNG générées
        png_files = sorted(list(layer_folder.glob("*.png")))
        
        for png_file in tqdm(png_files, desc=f"Calcul MSE pour {layer_folder.name}"):
            # Extraire le numéro d'image (ex: 1_layers_im0.png -> 0)
            # Flexible pour différents formats de nommage
            stem = png_file.stem
            try:
                img_num = int(stem.split('_im')[-1])
            except (ValueError, IndexError):
                # Essayer d'autres formats
                try:
                    img_num = int(''.join(filter(str.isdigit, stem)))
                except ValueError:
                    print(f"Impossible d'extraire le numéro d'image de {png_file.name}")
                    continue
            
            # Charger l'image générée
            generated_img = load_png_image(png_file)
            
            # Trouver l'image originale correspondante
            # Parcourir tous les cas dans cropped_centered
            found = False
            for case_folder in sorted(cropped_path.iterdir()):
                if not case_folder.is_dir():
                    continue
                
                slice_image_dir = case_folder / "Slice" / "Image"
                if not slice_image_dir.exists():
                    continue
                
                # Compter les images disponibles dans ce cas
                npy_files = sorted(list(slice_image_dir.glob("slice_*.npy")))
                
                if img_num < len(npy_files):
                    # Charger l'image originale
                    original_img = load_npy_image(npy_files[img_num])
                    
                    # Calculer la MSE
                    mse_value = calculate_mse(original_img, generated_img)
                    
                    results.append({
                        'attention_layer': layer_folder.name,
                        'generated_image': png_file.name,
                        'image_number': img_num,
                        'original_case': case_folder.name,
                        'original_image': npy_files[img_num].name,
                        'mse': mse_value
                    })
                    
                    found = True
                    break
            
            if not found:
                print(f"Attention: Image originale non trouvée pour {png_file.name}")
    
    # Créer un DataFrame et sauvegarder les résultats
    df = pd.DataFrame(results)
    
    if df.empty:
        print("Aucune paire d'images trouvée pour calculer la MSE.")
        return None
    
    # Afficher les statistiques pour chaque modèle (dossier) séparément
    print("\n" + "="*80)
    print("RÉSULTATS MSE PAR MODÈLE")
    print("="*80)
    
    # Sauvegarder les résultats séparément pour chaque modèle
    all_summaries = []
    
    for idx, layer in enumerate(sorted(df['attention_layer'].unique()), start=1):
        layer_data = df[df['attention_layer'] == layer]
        
        if not layer_data.empty:
            print(f"\n{layer}:")
            print(f"  Nombre d'images comparées: {len(layer_data)}")
            print(f"  MSE moyenne: {layer_data['mse'].mean():.6f}")
            print(f"  MSE médiane: {layer_data['mse'].median():.6f}")
            print(f"  MSE min: {layer_data['mse'].min():.6f}")
            print(f"  MSE max: {layer_data['mse'].max():.6f}")
            print(f"  Écart-type: {layer_data['mse'].std():.6f}")
            
            # Sauvegarder les résultats détaillés pour ce modèle
            output_file = Path(f"mse_results_{idx}.csv")
            layer_data.to_csv(output_file, index=False)
            print(f"  ✓ Résultats sauvegardés dans: {output_file}")
            
            # Ajouter au résumé global
            all_summaries.append({
                'model': layer,
                'count': len(layer_data),
                'mean': layer_data['mse'].mean(),
                'median': layer_data['mse'].median(),
                'std': layer_data['mse'].std(),
                'min': layer_data['mse'].min(),
                'max': layer_data['mse'].max()
            })
    
    # Sauvegarder le résumé global
    summary_df = pd.DataFrame(all_summaries)
    summary_file = Path("mse_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Résumé global sauvegardé dans: {summary_file}")
    
    return df


if __name__ == "__main__":
    try : 
        df_results = main()
    except Exception as e:
        print( f"got error : {e}")    
    
    duration = time() - start
    with open("time_use.log",'a') as f:
        f.write(f"{datetime.now()}|{duration}\n")
        print("durée enregistrée")

    