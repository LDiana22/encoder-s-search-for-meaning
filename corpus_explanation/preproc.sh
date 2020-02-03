#!/bin/bash
#SBATCH --time=01:45:00
#SBATCH --mail-user=dluca058@uottawa.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

#SBATCH --job-name=preprocessing

module load python/3.6



source env/bin/activate
python main.py
