import nibabel as nib
import numpy as np
import sys

def get_bounding_boxes(volume, label=None):
    """
    Retourne les bounding boxes d'un volume 3D pour une étiquette donnée (ou toutes les régions non nulles).
    
    Args:
        volume (numpy.ndarray): tableau 3D du volume.
        label (int, optional): valeur de l'étiquette. Si None, cherche toutes les valeurs > 0.
    
    Returns:
        list of dict: liste de bounding boxes, chaque bbox sous forme de dictionnaire.
    """
    if label is not None:
        mask = volume == label
        if not np.any(mask):
            print(f"Aucune région trouvée pour le label {label}")
            return []
        labels = [label]
    else:
        labels = np.unique(volume)
        labels = labels[labels != 0]  # ignore background

    bboxes = []
    for lbl in labels:
        coords = np.argwhere(volume == lbl)
        if coords.size == 0:
            continue
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        bboxes.append({
            "label": int(lbl),
            "xmin": int(xmin),
            "xmax": int(xmax),
            "ymin": int(ymin),
            "ymax": int(ymax),
            "zmin": int(zmin),
            "zmax": int(zmax),
            "shape": (xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1)
        })
    return bboxes


def main(nii_path, label=None):

    nii = nib.load(nii_path)
    volume = nii.get_fdata()
    volume = np.asarray(volume)
    bboxes = get_bounding_boxes(volume, label)

    print(f"\n=== Bounding boxes trouvées dans {nii_path} ===")
    if not bboxes:
        print("Aucune région détectée.")
        return

    for i, bbox in enumerate(bboxes):
        print(f"\nBBox {i+1}:")
        for k, v in bbox.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python niigz.py volume.nii.gz [label]")
        sys.exit(1)

    nii_path = sys.argv[1]
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(nii_path, label)
