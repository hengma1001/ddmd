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
To be added. 

## Run Workflow.
The workflow contains 3 main components, Molecular Dynamics simulations, machine learning neural network and inference node. This guild will provide instruction to set up your own system. 

1. MD simulations (`MD_exps`)

    The MD simulations requires inputs of molecular positions and interactions. For the moment, these files are ported to workflow in line 30-34 of `run_local.py` file. 

2. Agglomerating thread (`MD_to_CVAE`)
    
    This thread works as a watcher for the MD simulation process. In the beginning of the workflow, it withholds the ML training until sufficient MD simulations frames are collects. Later on, it manages retraining of ML network, whenever it accumulates 1.6 times more frames than previous training. 

    The number of the MD frames to initiate the first ML training can be modified in line 38 of `run_local.py`. 


3. ML network (`CVAE_exps`)

    The workflow currently uses a convolutional variational autoencoder to embed the bio-molecules into a low-dimensional latent representations. The `batch_size` can be modified in line 42 of `run_local.py`. 

    More ML architecture parameters are stored in `CVAE_exps/cvae/CVAE.py`. For bigger or more complex systems, the network could be too big for the available hardware to the user. Some adjustments can be made to the CVAE architecture to accommodate the challenge if simply changing batch size is insufficient. The common practice was increasing the stride size or reducing the filter depth of ConV layers. 

2. Inference (`Outlier_search`)
   
   The inference between MD and ML is managed by `Outlier_search/outlier_locator.py`. The current version searches the latent space for outliers and restarts MD simulations from these outliers. It also trims out unproductive MD simulations to make resources available. By default, it picks top 100 outliers and ranks them according to their local outlier factors. The top one will be used to start the next new MD simulation. 

   The number of outliers can be changed by `--n_out` to the execution script in line 128 of `run_local.py`. 


## Development notes
1. The current inputs and parameters assignments are hard-coded in the running script. A formatted input, such as `yml`, should be in place soon. 
