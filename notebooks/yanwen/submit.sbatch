#!/bin/bash

#
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --ntasks=16
#SBATCH --partition milano --account lcls:xppl1001021 
mpirun -n 16 python mpi_jungfrau.py
