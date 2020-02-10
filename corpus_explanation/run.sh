#!/bin/bash
#SBATCH --time=6:45:00
#SBATCH --mail-user=dluca058@uottawa.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=out/%x-%j.out
#SBATCH --job-name=nodes-6h

module load python/3.6



source env/bin/activate
python main.py
