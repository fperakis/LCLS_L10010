#!/bin/bash
#
#SBATCH --job-name=static# Job name for allocation
#SBATCH --output=/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/logs/contrast_analysis/%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/logs/contrast_analysis/%j.error # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=milano # Partition/Queue to submit job
#SBATCH --ntasks=1 # Total number of tasks
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --mail-type=ALL # Type of e-mail from slurm; other options are: Error, Info.
#SBATCH --reservation=lcls:onshift
#SBATCH --account=lcls:xppl1001021  

python analysis_epix5.py $1