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
pip install .
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
ddmd run_ddmd -c simple.yml
```
### Customized Run
A detailed setup for BBA in implicit solvent can be [found here](examples/example_imp.yml) and an example for explicit is also [available](examples/example_exp.yml). 

## Workflow parameters
The workflow is set up with a yaml file, as in [`example_imp.yml`](examples/example_imp.yml) or [`example_exp.yml`](examples/example_exp.yml), where all the available options/nobs are listed. 

- title: title for the workflow
- continue: `True` or `False`, whether to keep the previous run exists in the `output_dir`. It's still under development, as the unfinished MD runs from previous run will also be included. It's recommand to remove them manually prior to using this option. 
- conda_env: `conda env path`, conda environment where all the dependencies are installed. It will pick up the current shell conda env if unspecified. 
- n_sims: `int`, number of simulations to run. It will automatically be trimmed if not enough resource/GPUs present. 
- output_dir: `output path`, output directory. Highly recommand to use SSDs. 

- md_setup: the nested setup for md simulations 
  - pdb_file: path to input pdb 
  - top_file: path to topology file 
  - checkpoint: OpenMM checkpoint if continuing a run 
  - sim_time: length of MD simulation section, in ns 
  - report_time: trajectory and log output frequency, in ps 
  - dt: MD simulations time step, in fs
  - explicit_sol: `Bool` describing the solvent condition 
  - temperature: simulation temperture, in K 
  - pressure: simulation pressure, in bar
  - nonbonded_cutoff: cutoff for nonbonded interactions, in nm
  - init_vel: `Bool`, whether to initial atom velocity starting a simualtion 
  - max_iter: the maximum iterations of MD simulation sections to run

- ml_setup: the nested yaml for ml setup
  - n_train_start: number of frames to start cvae training 
  - reatrain_freq: `1-` how much data the workflow needs to start retraining, `retrain_freq` times of the previous training 
  - batch_size: training batch size  
  - epochs: training epochs
  - latent_dim: latent dimension of vae 
  - n_conv_layers: number of convolutional layers 
  - feature_maps: list of feature map depth for the conv layers 
  - filter_shapes: list of feature map shapes for the conv layers
  - strides: strides for each conv layers 
  - dense_layers: number of dense layers 
  - dense_neurons: number of neurons for dense layers 
  - dense_dropouts: dropout rate of dense layers 
  - atom_sel: selection string for building contact maps 
  - cutoff: cutoff for contact maps in angstrom 

- infer_setup: nested setup for inference setup 
  - n_outliers: number of outliers to identify 
  - md_threshold: how long a simulation needs to run before being qualified to stop 
  - screen_iter: output logs of outliers every 10 iteration
  - ref_pdb: target pdb for folding, also used to calculate rmsd 
  - atom_sel: selection string for rmsd calculation 
  - n_neighbors: number of neighbors to consider when calculating LOF
  - other LOF setup can also be ported here

## Analysis
The analysis of the simulation runs can be done through the following command. 
```bash
ddmd analysis -c infer.yml
```
It generates two pickle files, one of which contains all the simulation results, `result.pkl` and the other with the sampling efficiency information, `sampling.pkl`. 