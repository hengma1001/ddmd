import os
import glob
from tempfile import NamedTemporaryFile
from launcher import (
    ComputeNodeManager, 
    InsufficientResources,
    MPIRun,
)
import time

# Commands that run in a sub-shell immediately before each task:
MPIRun.set_preamble_commands(
    'eval "$(/lus/theta-fs0/projects/RL-fold/msalim/miniconda3/bin/conda shell.bash hook)"',
    'conda activate /lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm',
)


md_path = os.path.abspath("./MD_exps") 
input_paths = glob.glob('/lus/theta-fs0/projects/RL-fold/hengma/nsp10_16/amber_setup/input_*')
n_sim = len(input_paths) 

node_manager = ComputeNodeManager()
num_nodes = len(node_manager.nodes)

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
    output_file = NamedTemporaryFile(dir="./test-outputs", delete=False)

    run = MPIRun(
        cmd_line=md_cmd,
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpus,
        output_file=output_file,
        cwd="./MD_exps", # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run)

# 2 dual-GPU runs
# for i in range(2):
#     nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=2)
# 
#     output_file = NamedTemporaryFile(dir="./test-outputs", delete=False)
# 
#     run = MPIRun(
#         cmd_line=app_cmd,
#         node_list=nodes,
#         ranks_per_node=1,
#         gpu_ids=gpus,
#         output_file=output_file,
#         cwd=None, # can be a different working directory
#         envs_dict=None, # can be a dictionary of environ vars to add
#     )
#     runs.append(run)

while runs:
    # Clean up runs which finished (poll() returns process code)
    print("waiting on", len(runs), "runs to finish...")
    runs = [run for run in runs if run.poll() is None]
    time.sleep(5)

print("All done!")
