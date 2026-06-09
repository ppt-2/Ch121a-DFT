#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
module load orca/6.1.0 
pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history

path\ to\ executable\ of\ orca test.inp > test.out


date

