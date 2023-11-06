#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"): 
	Script to launch a smalldata_tools run analysis
	
	OPTIONS:
        -h|--help
            Definition of options
        -q|--queue
            Queue to use on SLURM
        -c|--cores
            Number of cores to be utilized
        -r|--run
            Run Number
EOF

}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
    -h|--help)
        usage
        exit
        ;;
    -q|--queue)
        QUEUE="$2"
        shift
        shift
        ;;
    -c|--cores)
        CORES="$2"
        shift
        shift
        ;;
    -r|--run)
        RUN=$2
        POSITIONAL+=("--run $2")
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


RUN="${RUN_NUM:=$RUN}" # default to the environment variable if submitted from the elog
# Export RUN for when running form the CLI
# This should just re-write the existing variables when running from the elog
export RUN_NUM=$RUN

DEFQUEUE='milano'
QUEUE=${QUEUE:=$DEFQUEUE}
CORES=${CORES:=120} # a full node by default
MAX_NODES=1

ABS_PATH='/sdf/data/lcls/ds/xpp/xppl1001021/results/shared'
PYTHONEXE=epix5_indiv_img.py
LOGDIR='/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/epix5_img/log'
LOGFILE=${LOGDIR}'/save_epix5_img_Run'${RUN_NUM}'_%J.log'

ACCOUNT='lcls:xppl1001021'
RESERVATION='lcls:onshift'

# for LCLS-I experiment on S3DF
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
echo source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

SBATCH_ARGS="-p $QUEUE --nodes 0-$MAX_NODES --ntasks $CORES --output $LOGFILE --reservation $RESERVATION"
MPI_CMD="mpirun -np $CORES python -u -m mpi4py.run ${ABS_PATH}/${PYTHONEXE} $*"

echo ---- $ABS_PATH/$PYTHONEXE $@
echo $SBATCH_ARGS --account $ACCOUNT --wrap="$MPI_CMD"
sbatch $SBATCH_ARGS --account $ACCOUNT --wrap="$MPI_CMD"

# #
# #SBATCH --job-name=test
# #SBATCH --output=output-%j.txt
# #SBATCH --error=output-%j.txt
# #
# #SBATCH --ntasks=16
# #SBATCH --partition milano --account lcls:xppl1001021 
# mpirun -n 16 python fast_Iq_intg.py
