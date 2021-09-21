import os
import time
import glob
import atexit
import GPUtil
import logging
from tempfile import NamedTemporaryFile
from launcher import (
    GPUManager, 
    MPIRun,
)

debug = 1
logger_level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

gpu_manager = GPUManager()

# conda env
conda_path = "/homes/heng.ma/miniconda3/envs/ddmd" 

# # Commands that run in a sub-shell immediately before each task:
# MPIRun.set_preamble_commands(
#     'eval "$(/home/hengma/miniconda3/bin/conda shell.bash hook)"',
#     f'conda activate {conda_path}',
# )

python_exe = f'{conda_path}/bin/python'

# MD setup 
md_path = os.path.abspath("./MD_exps") 
pdb_file = md_path + "/pdb/prot.pdb"
top_file = md_path + "/pdb/prot.prmtop"
sim_length = 20

# collecter setup 
collect_path = os.path.abspath("./MD_to_CVAE/")
n_frame_1 = 1000

# train setup 
train_path = os.path.abspath("./CVAE_exps") 
batch_size = 128

# inference setup 
inf_path = os.path.abspath("./Outlier_search") 

# run setup 
n_sim = 4
n_train = 1 
n_gpus = n_sim + n_train + 1 
logger.info(f"Configuration: {n_sim} MDs, "\
        f"{n_train} training, and 1 inference node. "\
        f"Need {n_gpus} GPUs. ")

runs = []
os.makedirs("test-outputs", exist_ok=True)

gpu_ids = gpu_manager.request(n_gpus) 
logger.info(f"Available {len(gpu_ids)} GPUs: {gpu_ids}")
if len(gpu_ids) < n_gpus: 
    n_sim = n_sim - (n_gpus - len(gpu_ids)) 
    n_gpus = len(gpu_ids) 
    logger.info("Not enough GPUs available, reduce "\
            f"number of simulations to {n_sim}. ")
    logger.info(f"New configuration: {n_sim} MDs, "\
            f"{n_train} training, and 1 inference node. "\
            f"Need {n_gpus} GPUs. ")

# Single GPU MD Runs
for i in range(n_sim):
    gpus = [gpu_ids.pop()]

    # input_path = input_paths[i]
    # sys_label = os.path.basename(input_path).replace('input_', '')
    # pdb_file = input_path + f"/{sys_label}.pdb"
    # top_file = input_path + f"/{sys_label}.top"
    md_cmd = (
            f"python run_openmm.py -f {pdb_file} -p {top_file} -l {sim_length}"
    )
    output_file = f"./test-outputs/MD_{i}"

    run = MPIRun(
        cmd_line=md_cmd,
        num_ranks=1,
        gpu_ids=gpus,
        output_file=output_file,
        cwd=md_path, # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run)

# collecting phase
# python MD_to_CVAE.py -f ../MD_exps/ -l 1000
col_cmd = f"{python_exe} MD_to_CVAE.py -f {md_path} -l {n_frame_1}"
output_file = "./test-outputs" + "/collect_run"
run = MPIRun(
    cmd_line=col_cmd,
    num_ranks=1,
    output_file=output_file,
    cwd=collect_path, # can be a different working directory
    envs_dict=None, # can be a dictionary of environ vars to add
)
runs.append(run)

# training phase 
# python train_cvae.py -f ../MD_to_CVAE/cvae_input.h5 -b 100  -d 4
for i in range(n_train): 
    dim = i + 3 
    gpus = gpu_ids.pop()
    train_cmd = f"python train_cvae.py "\
                f"-f {collect_path}/cvae_input.h5 "\
                f"-d {dim} -g {gpus} -b {batch_size}" 
    output_file = "./test-outputs" + f"/train_{i}"
    run = MPIRun(
        cmd_line=train_cmd,
        num_ranks=1,
        gpu_ids=[gpus],
        output_file=output_file,
        cwd=train_path, # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run) 

# inferencing node 
# export PYTHONPATH=$PYTHONPATH:/autofs/nccs-svm1_home1/hm0/med110_proj/entk_cvae_md/CVAE_exps
# jsrun -n 1 -a 1 -g 1 python outlier_locator.py -m ../MD_exps/ -c ../CVAE_exps/ -p ../MD_exps/pdb/fs-peptide.pdb -r ../MD_exps/pdb/fs-peptide.pdb
gpus = gpu_ids.pop()
inf_cmd = f"python outlier_locator.py "\
          f"-m {md_path} -c {train_path} "\
          f"-p {pdb_file} -g {gpus}" 
output_file = "./test-outputs" + "/inference_output" 
env_dict={"PYTHONPATH": train_path}
run = MPIRun(
        cmd_line=inf_cmd, 
        num_ranks=1, 
        gpu_ids=[gpus], 
        output_file=output_file, 
        cwd=inf_path, 
        envs_dict=env_dict,
        )
runs.append(run) 

# set up cleanup at exiting program 
def cleanup(): 
    for p in runs: 
        p.kill()
    logger.info("cleaned up!") 

atexit.register(cleanup)

print("waiting on", len(runs), "runs to finish...")
while runs:
    # Clean up runs which finished (poll() returns process code)
    runnings = [run for run in runs if run.poll() is None]
    if len(runs) != len(runnings):
        print("waiting on", len(runnings), "runs to finish...")
        runs = runnings 
    time.sleep(5)

print("All done!")
