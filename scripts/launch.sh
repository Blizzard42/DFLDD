#!/bin/bash
#SBATCH --job-name=dfldd_%j
#SBATCH --output=./runs/logs/run_%j.out
#SBATCH --error=./runs/logs/run_%j.err
#SBATCH --partition=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --mem=0
#SBATCH --exclude="xaea-12"

export PYTHONIOENCODING=UTF-8
source ~/.bashrc
# export HF_HOME="$HOME/flash/huggingface"
# source /nethome/gpatlin3/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate dfldd
# cd $SLURM_SUBMIT_DIR
cd ~/flash/DFLDD

srun python -m src.cifar10
