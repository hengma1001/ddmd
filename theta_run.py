import os
import time
import logging
import glob
from tempfile import NamedTemporaryFile
from launcher import (
    ComputeNodeManager, 
    InsufficientResources,
    MPIRun,
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

node_manager = ComputeNodeManager()
n_nodes = len(node_manager.nodes)

# conda env
# conda_path = '/lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm'
conda_path = "/home/hengma/miniconda3/envs/A100_rapids_openmm" 

# Commands that run in a sub-shell immediately before each task:
MPIRun.set_preamble_commands(
    'eval "$(/home/hengma/miniconda3/bin/conda shell.bash hook)"',
    f'conda activate {conda_path}',
)

container_path = "/lus/theta-fs0/projects/RL-fold/hengma/cvae_md_wf/cvae_md.sif"
container_cmd = "singularity run -B /lus:/lus:rw -B /raid:/raid:rw "\
                f"--nv {container_path} "

python_exe = f'{conda_path}/bin/python'

# MD setup 
md_path = os.path.abspath("./MD_exps") 
pdb_file = md_path + "/pdb/100-fs-peptide-400K.pdb"
ref_pdb = md_path + "/pdb/fs-peptide.pdb"

# collecter setup 
collect_path = os.path.abspath("./MD_to_CVAE/")
n_frame_1 = 1000

# train setup 
train_path = os.path.abspath("./CVAE_exps") 

# inference setup 
inf_path = os.path.abspath("./Outlier_search") 

# run setup 
n_res = n_nodes * 8
n_train = 2 
n_sim = n_res - n_train - 1 

runs = []
os.makedirs("test-outputs", exist_ok=True)

# 4 Single GPU Runs
for i in range(n_sim):
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)

    input_path = input_paths[i]
    sys_label = os.path.basename(input_path).replace('input_', '')
    pdb_file = input_path + f"/{sys_label}.pdb"
    top_file = input_path + f"/{sys_label}.top"
    md_cmd = (
            f"python run_openmm.py -f {pdb_file} -p {top_file} -l 10"
    )
    output_file = f"./test-outputs/MD_{i}"

    run = MPIRun(
        cmd_line=md_cmd,
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpus,
        output_file=output_file,
        cwd=md_path, # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run)

# collecting phase
# python MD_to_CVAE.py -f ../MD_exps/ -l 1000
nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=0)
col_cmd = f"{python_exe} MD_to_CVAE.py -f {md_path} -l {n_frame_1}"
output_file = "./test-outputs" + "/collect_run"
run = MPIRun(
    cmd_line=col_cmd,
    node_list=nodes,
    ranks_per_node=1,
    gpu_ids=gpus,
    output_file=output_file,
    cwd=collect_path, # can be a different working directory
    envs_dict=None, # can be a dictionary of environ vars to add
)
runs.append(run)

# training phase 
# python train_cvae.py -f ../MD_to_CVAE/cvae_input.h5 -b 100  -d 4
for i in range(n_train): 
    dim = i + 3 
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
    train_cmd = f"{container_cmd} python train_cvae.py "\
                f"-f {collect_path}/cvae_input.h5 -d {dim} "\
                f"-g {','.join(str(x) for x in list(gpus))}"
    output_file = "./test-outputs" + f"/train_{i}"
    run = MPIRun(
        cmd_line=train_cmd,
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpus,
        output_file=output_file,
        cwd=train_path, # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run) 

# inferencing node 
# export PYTHONPATH=$PYTHONPATH:/autofs/nccs-svm1_home1/hm0/med110_proj/entk_cvae_md/CVAE_exps
# jsrun -n 1 -a 1 -g 1 python outlier_locator.py -m ../MD_exps/ -c ../CVAE_exps/ -p ../MD_exps/pdb/fs-peptide.pdb -r ../MD_exps/pdb/fs-peptide.pdb
nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
inf_cmd = f"{container_cmd} python outlier_locator.py "\
          f"-m {md_path} -c {train_path} "\
          f"-p {ref_pdb} -r {ref_pdb} "\
          f"-g {','.join(str(x) for x in list(gpus))}"
output_file = "./test-outputs" + "/inference_output" 
env_dict={"PYTHONPATH": train_path}
run = MPIRun(
        cmd_line=inf_cmd, 
        node_list=nodes, 
        ranks_per_node=1, 
        gpu_ids=gpus, 
        output_file=output_file, 
        cwd=inf_path, 
        envs_dict=env_dict,
        )
runs.append(run) 

while runs:
    # Clean up runs which finished (poll() returns process code)
    print("waiting on", len(runs), "runs to finish...")
    runs = [run for run in runs if run.poll() is None]
    time.sleep(5)

print("All done!")
