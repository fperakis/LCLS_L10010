#!/bin/bash

#SBATCH --job-name=avg_img_xtc# Job name for allocation
#SBATCH --output=%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=%j.error # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=milano # Partition/Queue to submit job
#SBATCH --ntasks=1 # Total number of tasks
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --reservation=lcls:onshift
#SBATCH --account=lcls:xppl1001021

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
    --run)
        RUN=$2
        shift
        shift
        ;;
    --experiment)
        EXP=$2
        shift
        shift
        ;;
    --nimgs)
        NIMGS=$2
        shift
        shift
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;                     
    esac
done
set -- "${POSITIONAL[@]}"

python3 /sdf/data/lcls/ds/xpp/xppl1001021/results/shared/avg_img_from_xtc.py --detector "epix_alc5" --run $RUN --experiment $EXP --nimgs $NIMGS