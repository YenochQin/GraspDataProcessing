#!/bin/zsh
#SBATCH -J mlci_rf
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH -p WorkStationOne
#SBATCH --output=mlci_rf.log
. /usr/share/Modules/init/zsh
# grasp2018 MCDHF MR calculation
module load mpi/openmpi-x86_64
export PATH=/home/workstation1/AppFiles/grasp/bin:$PATH

nohup python mlci_rf.py