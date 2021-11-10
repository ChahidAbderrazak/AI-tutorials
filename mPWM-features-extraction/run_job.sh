#!/bin/bash

#SBATCH --nodes=1

#SBATCH --output=output.out

#SBATCH --error=error_out.err

#SBATCH --time=60:00:00 

#SBATCH --partition=batch

#SBATCH --job-name=QuPWM

#SBATCH --cpus-per-task=8

#run the application:

module load anaconda3

conda env create --prefix ./env --file environment.yml

conda activate ./env

python main.py
