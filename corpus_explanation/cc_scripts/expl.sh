#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --mail-user=dluca058@uottawa.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

#SBATCH --job-name=dictionary
#SBATCH --output=out/%x-%j.out

module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env_c
source ../../env/bin/activate

python ../main.py
