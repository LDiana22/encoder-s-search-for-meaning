#!/bin/bash
#SBATCH --time=05:45:00
#SBATCH --mail-user=dluca058@uottawa.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --output=out/%x-%j.out
#SBATCH --job-name=vanila

module load python/3.6



source env/bin/activate
python main.py
