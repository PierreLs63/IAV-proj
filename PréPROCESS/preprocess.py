import os
import nibabel as nib
import numpy as np
import shutil

def crop_center_keep_depth(volume, target_xy=(64, 64), depth_axis=-1):
    """
    Crop centré sur les deux axes sauf celui de profondeur.
    - volume: ndarray 3D
    - target_xy: (height, width) pour les axes croppés
    - depth_axis: indice de l'axe de profondeur à conserver
    Retour: volume croppé, profondeur conservée
    """
    # Normaliser depth_axis
    depth_axis = depth_axis if depth_axis >= 0 else volume.ndim + depth_axis

    # Identifier les axes à croper
    axes = [0, 1, 2]
    crop_axes = [a for a in axes if a != depth_axis]

    # Permuter volume pour avoir (crop_axis0, crop_axis1, depth)
    perm = crop_axes + [depth_axis]
    vol_perm = np.transpose(volume, perm)
    A, B, D = vol_perm.shape  # A,B = axes croppées, D = profondeur

    # Centre des axes croppés
    ca, cb = A // 2, B // 2
    ty, tx = target_xy
    a1, a2 = max(0, ca - ty//2), min(A, ca + ty//2)
    b1, b2 = max(0, cb - tx//2), min(B, cb + tx//2)

    crop = vol_perm[a1:a2, b1:b2, :]
    # Padding si nécessaire
    pad_a = ty - crop.shape[0]
    pad_b = tx - crop.shape[1]
    pad_before_a = pad_a // 2
    pad_after_a = pad_a - pad_before_a
    pad_before_b = pad_b // 2
    pad_after_b = pad_b - pad_before_b

    crop = np.pad(crop,
                  ((pad_before_a, pad_after_a),
                   (pad_before_b, pad_after_b),
                   (0, 0)),
                  mode='constant',
                  constant_values=0)

    # Remettre l’ordre original des axes
    inv_perm = np.argsort(perm)
    crop_back = np.transpose(crop, inv_perm)
    return crop_back

def check_mask_extent(mask_data, patient_name):
    """Vérifie si le masque dépasse 60x60 en XY."""
    coords = np.argwhere(mask_data > 0)
    if coords.size == 0:
        return  # pas de labels
    # XY = axes 0 et 1 si profondeur est sur l'axe 2
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    width = x_coords.max() - x_coords.min() + 1
    height = y_coords.max() - y_coords.min() + 1
    if width > 60 or height > 60:
        print(f"!!!! [ALERTE] {patient_name} : masque dépasse {width}x{height} (limite 60x60)")

def process_patient(patient_dir, output_dir):
    """Traite un patient (crop centré image + masque + export slices)."""
    name = os.path.basename(patient_dir)

    # Chercher les fichiers .nii.gz dans Images et Contours
    img_folder = os.path.join(patient_dir, "Images")
    mask_folder = os.path.join(patient_dir, "Contours")

    img_files = [f for f in os.listdir(img_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

    if len(img_files) == 0 or len(mask_files) == 0:
        print(f"!!!! Fichiers manquants pour {name}")
        return

    # On prend le premier fichier trouvé
    img_path = os.path.join(img_folder, img_files[0])
    mask_path = os.path.join(mask_folder, mask_files[0])

    # Chargement des volumes
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # Remapper les labels 4 → 3
    mask_data[mask_data == 4] = 3

    # Vérification du masque avant cropping
    check_mask_extent(mask_data, name)

    # Crop centré
    img_crop = crop_center_keep_depth(img_data, target_xy=(64, 64), depth_axis=2)
    mask_crop = crop_center_keep_depth(mask_data, target_xy=(64, 64), depth_axis=2)

    # Création du dossier patient
    out_patient_dir = os.path.join(output_dir, name)
    os.makedirs(out_patient_dir, exist_ok=True)

    # Sauvegarde volumes NIfTI
    nib.save(nib.Nifti1Image(img_crop, affine=np.eye(4)),
             os.path.join(out_patient_dir, f"{name}_img.nii.gz"))
    nib.save(nib.Nifti1Image(mask_crop, affine=np.eye(4)),
             os.path.join(out_patient_dir, f"{name}_mask.nii.gz"))

    # 🔹 Création des sous-dossiers pour slices
    slice_img_dir = os.path.join(out_patient_dir, "Slice", "Image")
    slice_mask_dir = os.path.join(out_patient_dir, "Slice", "Mask")
    os.makedirs(slice_img_dir, exist_ok=True)
    os.makedirs(slice_mask_dir, exist_ok=True)

    # Sauvegarde chaque slice 2D
    num_slices = img_crop.shape[2]  # axe profondeur
    for i in range(num_slices):
        slice_img = img_crop[:, :, i]
        slice_mask = mask_crop[:, :, i]
        np.save(os.path.join(slice_img_dir, f"slice_{i:03d}.npy"), slice_img)
        np.save(os.path.join(slice_mask_dir, f"slice_{i:03d}.npy"), slice_mask)

    print(f"✅ {name} -> crop centré sauvegardé ({img_crop.shape}) + {num_slices} slices exportées")


def process_all_patients(root_dir, output_dir="cropped_centered"):
    """Traite tous les patients du dossier racine."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patients = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if d.startswith("Case_") and os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f"🔍 {len(patients)} patients trouvés dans {root_dir}")

    for p in patients:
        process_patient(p, output_dir)

    print("\n --- Traitement terminé. ---")
    print(f"Tous les volumes cropped sont dans : {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage : python preproceess.py dossier_racine [dossier_sortie]")
        sys.exit(1)

    root_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "cropped_centered"
    process_all_patients(root_dir, out_dir)
