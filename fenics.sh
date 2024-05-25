#!/bin/bash
#SBATCH -J 4particles  # Job name
#SBATCH -p cpu               # Partition (queue) name
#SBATCH -o %j.out            # Standard output log file name (using job ID)
#SBATCH -e %j.err            # Standard error log file name (using job ID)
#SBATCH --ntasks-per-node=1 # Number of MPI tasks (processes) per node
#SBATCH -N 1                 # Number of nodes requested


#主程序
export DIJITSO_CACHE_DIR=/home/daixiaoxu/fenics/particle/raidus60/more particles raidus60/chain/4p

#mpirun -np 10 --mca orte_base_help_aggregate 0 python3 solidSintering_11152023_MPI.py $(date +"%Y-%m-%d_%H-%M-%S")


python3 viscousSintering.py


