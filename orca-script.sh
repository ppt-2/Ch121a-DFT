#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
module load openmpi/4.1.2-gcc-11.3.1-7kosie3
pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history

/resnick/groups/wag/prabhat/programs/orca-6.0.0/orca_6_0_0/orca test.inp > test.out


date
