# DDMD

A script that runs Molecular Dynamics simulations under supervision of Machine
Learning model on local workstation.

Here is the step-by-step instruction for setting up the workflow.

## Environment setup

Use one of the following ways to set up a working environment for ddmd. 

### Conda environment 
The workflow requires various packages to perform different tasks, which can be easily acquired through [anaconda](https://www.anaconda.com/products/individual)/[miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager. 

After anaconda/miniconda is installed in your machine, the environment can be built simply by

```
conda env create -f envs/ddmd.yml
```

### Docker
Docker images are available and can be built with the following command. 
```
docker build -f envs/Dockerfile -t ddmd . 
docker run -it ddmd bash
```

### Singularity 
Singularity image can be built with `envs/ddmd.def`. 
```
cd envs
sudo singularity build ddmd.sif ddmd.def
```
Note: It would be necessary to specify the `singularity` path if it's not in the root dir. 


## Run Workflow

### Simple Run
The workflow can be simply run with example BBA folding implicit run. 
```bash 
cd examples
export PYTHONPATH=$PYTHONPATH:$(realpath ..)
python run_ddmd.py -c simple.yml
```
### Customized Run
A detailed setup for BBA in implicit solvent can be found [here](examples/example_imp.yml). 

