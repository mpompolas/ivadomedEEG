#!/bin/bash
#SBATCH --account=COMPUTE_CANADA_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu 4GB 
#SBATCH --time=04:00:00
#SBATCH --job-name=IvadoMedEEG
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=USER_EMAIL
#SBATCH --mail-type=ALL

cd PATH_TO_SCRIPT
module load StdEnv/2020
module load python/3.7
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
python SCRIPT.py