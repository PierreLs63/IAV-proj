#!/bin/bash
#SBATCH --account=m25172
#SBATCH --partition=mesonet
#SBATCH --job-name=training_genheart # nom du job
#SBATCH --time=00:00:30              # temps maximal d’allocation
#SBATCH --output=genheart.out      # nom du fichier de sortie
#SBATCH --error=genheart.err       # nom du fichier d'erreur (ici commun avec la sortie)

# Ici, reservation de 8 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=8            # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --mem=16G
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH -C v100-16g

module load Apptainer/1.3.6-1.el9
 
# Execution du code

apptainer exec --bind $PROJ_HOME:$PROJ_HOME --nv $PROJ_HOME/containers/pytorch_25.09-py3.sif 
python3 $PROJ_HOME/test_net.py

