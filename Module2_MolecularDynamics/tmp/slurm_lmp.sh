#!/bin/bash
#SBATCH --job-name=
##SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=pptkavi@gmail.com     # Where to send mail
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
#SBATCH --output=Logfile   # Standard output and error log
pwd; hostname; date


module load openmpi/5.0.1-gcc-11.3.1-j4o6ryt

pwd
cwd=$(pwd)
echo $SLURM_JOBID "${cwd}" >> ~/job_history

mpirun -np 4  /resnick/groups/wag/wag_programs/lammps-22Jul2025/src/lmp_mpi -in in.min_eql_prod


date

