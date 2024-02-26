#!/bin/bash
#
#SBATCH --job-name=static# Job name for allocation
#SBATCH --array=0-3
#SBATCH --output=logs/%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=logs/%j.error # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=milano # Partition/Queue to submit job
#SBATCH --ntasks=1 # Total number of tasks
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --mail-type=ALL # Type of e-mail from slurm; other options are: Error, Info.
#SBATCH --reservation=lcls:onshift
#SBATCH --account=lcls:xppl1001021  

python analysis_4epix.py $1