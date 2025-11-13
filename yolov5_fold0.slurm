#!/bin/bash
#SBATCH --account=xhu@v100
#SBATCH --chdir=logs_yolov5/
#SBATCH --job-name=trainingYolov5        # nom du job
#SBATCH --qos=qos_gpu-t4 # 100h max and 96 gpu nodes
#SBATCH --time=100:00:00              # temps maximal d’allocation
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --constraint v100-32g  # there is a small number of 16Go GPU

# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH -C v100-16g

 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
#module load cpuarch/amd
 
# Chargement des modules
module load pytorch-gpu/py3/2.0.1
conda deactivate
conda activate yolov5

export GIT_PYTHON_REFRESH=quiet
# Params

# Echo des commandes lancees
set -x

crop=True
 
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code

cd $WORK/yolov5
fold=0

sleep $((RANDOM % 100))

# 3 classes :
if [ $crop == "True" ]; then
    # Pay attention to the image size expectations of the albumentation package ... (utils/augmentations)
    python3 -u train.py --resume --data /gpfswork/rech/xhu/urb41hw/yolov5/data/conf_yolov5_crop_multimodal_fold$fold.yaml --hyp fold$fold/hyp.prostate_crop.yaml --weights yolov5m.pt --cfg yolov5-p2.yaml --img 96 --epochs 35 --batch 16 --optimizer AdamW --image-weights --evolve 320 --cache --name gridsearch_crop_fold0
else 
    python3 -u train.py --data /gpfswork/rech/xhu/urb41hw/yolov5/data/conf_yolov5_multimodal_fold$fold.yaml --hyp fold$fold/hyp.prostate.yaml --weights yolov5m.pt --cfg yolov5-p2.yaml --img 256 --epochs 45 --batch 16 --optimizer AdamW --image-weights --cache --name expfold0
fi
