JOB_NAME=$1
NUM_GPUS=$2

if [ -z "$JOB_NAME" ] || [ -z "$NUM_GPUS" ]; then
    echo "Usage: $0 <JOB_NAME> <NUM_GPUS>"
    exit 1
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=./runs/dec/${JOB_NAME}_%j.out
#SBATCH --error=./runs/dec/${JOB_NAME}_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a40:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --qos="short"
#SBATCH --partition=kira-lab
#SBATCH --signal=USR1@100
#SBATCH --exclude=xaea-12,puma
# #SBATCH --exclude=chomps,ephemeral-3,walle,friday,cyborg,starrysky,hk47,jill,xaea-12,johnny5,calculon,kitt,megazord,randotron,megabot,robby

# Load environment
source ~/.bashrc
conda deactivate
conda activate dfldd

echo "Starting dfldd job ${JOB_NAME}"

export WORLD_SIZE=\$(nvidia-smi -L | wc -l)
deepspeed --num_gpus \$WORLD_SIZE lm_deepspeed.py --learning_rate 1e-4

echo "dfldd job ${JOB_NAME} has finished"
EOF
