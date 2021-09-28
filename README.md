# DDMD
A script that runs Molecular Dynamics simulations under supervision of Machine 
Learning model on local workstation. 

## Environment setup 
The conda environment can be built on local machine via 
```
conda env create -f ddmd.yml
```
.

It should create a environment accommodates all workflow dependencies. 

## Run Workflow. 

1. Set up the input. 
    The system needs input in the same format of MD simulations, pdb and 
    topology file. Right file should be assigned in `run_local.py`. 
The system is ready to run at this point, except if the system size 
is too big for the current CVAE setup. 
2. Adjust CVAE parameter
    The CVAE parameters can be found in `CVAE_exps/cvae/CVAE.py`. 
