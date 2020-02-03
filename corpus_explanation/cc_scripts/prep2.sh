#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --mail-user=dluca058@uottawa.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

#SBATCH --job-name=prep
#SBATCH --output=out/%x-%j.out

module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env_c
source ../env/bin/activate
ls
pwd
python ../main2.py
