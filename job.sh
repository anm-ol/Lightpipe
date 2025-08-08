#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --partition=long
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=06:00:00

# Add your commands here
# For example:
blenderproc run src/main.py --config_path configs/configv1.yaml