#!/bin/bash
#SBATCH -J GBDF2_Simulation  # Job name
#SBATCH -p cpu               # Partition (queue) name
#SBATCH -o %j.out            # Standard output log file name (using job ID)
#SBATCH -e %j.err            # Standard error log file name (using job ID)
#SBATCH --ntasks-per-node=1 # Number of MPI tasks (processes) per node
#SBATCH -N 1                 # Number of nodes requested


#主程序
export DIJITSO_CACHE_DIR=/home/qianbo/Fenics/Code_Python_cluster/Viscosity_Sintering

#mpirun -np 10 --mca orte_base_help_aggregate 0 python3 solidSintering_11152023_MPI.py $(date +"%Y-%m-%d_%H-%M-%S")


python3 Viscosity_PolyParticle_BDF2.py