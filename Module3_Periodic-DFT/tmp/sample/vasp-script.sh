#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:30:00               # Time limit hrs:min:sec


module load intel-oneapi-mkl/2023.2.0-gcc-11.3.1-idewa6m intel-oneapi-compilers/2023.2.1-gcc-11.3.1-xbceufh intel-oneapi-mpi/2021.11.0-oneapi-2023.2.1-atn5hhn
pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history

mpirun -np 2 /resnick/groups/wag/programs/vasp.6.6.0/bin/vasp_std


date

