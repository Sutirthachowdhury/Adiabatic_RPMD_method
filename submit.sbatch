#!/bin/bash
#SBATCH --partition=et3
#SBATCH --output adia.out
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1


time python3 rpmd.py
