#!/bin/bash -x
#COBALT -A RL-FOLD
#COBALT -n 1
#COBALT -q default
#COBALT -t 60
#COBALT --attrs pubnet=true:enable_ssh=1:ssds=required:ssd_size=2048# Set up conda and activate Venkat's environment

export PATH=/home/hengma/miniconda3/bin:$PATH
which python 
python theta_run.py
# eval "$(/home/braceal/miniconda3/bin/conda shell.bash hook)"
# conda activate /lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm# Set env to SSH private key for accessing medulla
# python run_openmm.py -f pdb/100-fs-peptide-400K.pdb l 50 
