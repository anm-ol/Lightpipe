#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --partition=long
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log

# Print SLURM environment variables for debugging
echo "=== SLURM Environment ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "========================="

# Change to the working directory
cd /mnt/venky/ankitd/anmol/new_vace_training/Lightpipe

# Run the pipeline - each task will process a subset of videos
srun blenderproc run src/main.py --config configs/configv1.yaml
