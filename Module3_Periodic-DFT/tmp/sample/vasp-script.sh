#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:30:00               # Time limit hrs:min:sec


module load "load arch specific modules|"
pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history
ulimit -s unlimited
mpirun -np 8 path\ to\ vasp_std


date

