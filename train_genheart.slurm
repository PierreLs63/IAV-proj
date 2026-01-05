#!/bin/bash
#SBATCH --account=m25172
#SBATCH --partition=mesonet
#SBATCH --job-name=training_genheart # nom du job
#SBATCH --time=15:00:00              # temps maximal d’allocation
#SBATCH --output=genheart.out      # nom du fichier de sortie
#SBATCH --error=genheart.err       # nom du fichier d'erreur (ici commun avec la sortie)

# Ici, reservation de 8 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=8            # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --mem=16G
 
apptainer exec \
    --bind $PROJ_HOME:$PROJ_HOME \
    --nv /home/pipeyrie/IAV-proj/containers/pytorch_25.09-py3.sif \
    python3 /home/pipeyrie/IAV-proj/mask_diffusion_cfg.py


