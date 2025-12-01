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
    img = img.astype(np.float32)
    # Normaliser entre 0 et 1 en utilisant min-max normalization
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return torch.from_numpy(img).float()


def load_png_image(png_path):
    """Charge une image PNG et la convertit en tenseur PyTorch."""
    img = Image.open(png_path).convert('L')  # Convertir en niveaux de gris
    img = np.array(img, dtype=np.float32)
    # Normaliser entre 0 et 1
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
    
    # Indexer toutes les images originales par cas
    print("\nIndexation des images originales par cas...")
    cases_images = {}
    total_original = 0
    for case_folder in sorted(cropped_path.iterdir()):
        if not case_folder.is_dir():
            continue
        
        slice_image_dir = case_folder / "Slice" / "Image"
        if not slice_image_dir.exists():
            continue
        
        npy_files = sorted(list(slice_image_dir.glob("slice_*.npy")))
        if npy_files:
            cases_images[case_folder.name] = npy_files
            total_original += len(npy_files)
    
    print(f"Total de {len(cases_images)} cas avec {total_original} images originales")
    
    results = []
    
    # Parcourir chaque dossier de samples trouvé
    for layer_folder in sample_folders:
        print(f"\nTraitement de {layer_folder.name}...")
        
        # Récupérer toutes les images PNG générées
        png_files = sorted(list(layer_folder.glob("*.png")))
        
        print(f"Comparaison de {len(png_files)} images générées avec {total_original} images originales...")
        total_comparisons = len(png_files) * total_original
        print(f"Total de comparaisons à effectuer: {total_comparisons}")
        
        # Barre de progression pour toutes les comparaisons
        with tqdm(total=total_comparisons, desc=f"MSE {layer_folder.name}") as pbar:
            for png_file in png_files:
                # Charger l'image générée une seule fois
                generated_img = load_png_image(png_file)
                
                # Comparer avec chaque image originale de chaque cas
                for case_name, npy_files in cases_images.items():
                    for npy_file in npy_files:
                        # Charger l'image originale
                        original_img = load_npy_image(npy_file)
                        
                        # Calculer la MSE
                        mse_value = calculate_mse(original_img, generated_img)
                        
                        results.append({
                            'model': layer_folder.name,
                            'generated_image': png_file.name,
                            'original_case': case_name,
                            'original_image': npy_file.name,
                            'mse': mse_value
                        })
                        
                        pbar.update(1)
    
    # Créer un DataFrame et sauvegarder les résultats
    df = pd.DataFrame(results)
    
    if df.empty:
        print("Aucune paire d'images trouvée pour calculer la MSE.")
        return None
    
    print(f"\n✓ Total de {len(df)} comparaisons effectuées")
    
    # Afficher les statistiques pour chaque modèle
    print("\n" + "="*80)
    print("RÉSULTATS MSE PAR MODÈLE")
    print("="*80)
    
    # Sauvegarder les résultats séparément pour chaque modèle
    all_summaries = []
    
    for idx, model in enumerate(sorted(df['model'].unique()), start=1):
        model_data = df[df['model'] == model]
        
        if not model_data.empty:
            print(f"\n{model}:")
            print(f"  Nombre de comparaisons: {len(model_data)}")
            print(f"  MSE moyenne: {model_data['mse'].mean():.6f}")
            print(f"  MSE médiane: {model_data['mse'].median():.6f}")
            print(f"  MSE min: {model_data['mse'].min():.6f}")
            print(f"  MSE max: {model_data['mse'].max():.6f}")
            print(f"  Écart-type: {model_data['mse'].std():.6f}")
            
            # Sauvegarder les résultats détaillés pour ce modèle
            output_file = Path(f"mse_results_{idx}.csv")
            model_data.to_csv(output_file, index=False)
            print(f"  ✓ Résultats sauvegardés dans: {output_file}")
            
            # Ajouter au résumé global
            all_summaries.append({
                'model': model,
                'comparisons': len(model_data),
                'mean': model_data['mse'].mean(),
                'median': model_data['mse'].median(),
                'std': model_data['mse'].std(),
                'min': model_data['mse'].min(),
                'max': model_data['mse'].max()
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

    