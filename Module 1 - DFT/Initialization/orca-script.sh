#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
module load orca/6.1.0 
pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history

/central/software9/external/orca/orca_6_1_0_linux_x86-64_shared_openmpi418_avx2/orca test.inp > test.out


date

