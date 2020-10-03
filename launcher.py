from itertools import combinations
from typing import List, Dict, Tuple, Set
import os
import getpass
import subprocess
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


IDENTITY_FILE = os.environ.setdefault(
    "MEDULLA_IDENTITY_FILE", 
    os.path.expanduser('~/.ssh/id_rsa_medulla'),
)


class InsufficientResources(Exception): pass

class ComputeNode:
    def __init__(self, id, num_gpus=None):
        if num_gpus is None:
            num_gpus = int(os.environ.get("GPUS_PER_NODE", "8"))

        self.id = id
        self.idle_gpus = set(range(num_gpus))
        self.busy_gpus = set()

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"<ComputeNode: {self.id}>"
    
    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def get_node_list(cls, hostfile=None):
        if hostfile is None:
            hostfile = os.environ["COBALT_NODEFILE"]
        with open(hostfile) as fp:
            data = fp.read()
        splitter = ',' if ',' in data else None
        return [cls(node_id) for node_id in data.split(splitter)]
    
    def free_gpus(self, gpu_list: List[int]):
        for id in gpu_list:
            self.busy_gpus.remove(id)
            self.idle_gpus.add(id)

class ComputeNodeManager:
    def __init__(self, hostfile=None):
        self.nodes = ComputeNode.get_node_list(hostfile=hostfile)
        logger.info(f"Manager detected {len(self.nodes)} compute nodes")

    def request(self, num_nodes: int, gpus_per_node: int) -> Tuple[List[ComputeNode], Set[int]]:
        available_nodes = [
            node for node in self.nodes
            if len(node.idle_gpus) >= gpus_per_node
        ]
        idle_gpu_map = {
            node: node.idle_gpus
            for node in available_nodes
        }
        idle_gpu_sets = list(idle_gpu_map.values())

        if len(idle_gpu_map) < num_nodes:
            raise InsufficientResources("Not enough nodes have `gpus_per_node` free GPUs")

        # Select a common tuple of GPU IDs that is available on all num_nodes
        common_gpu_ids = (
            set(gids)
            for gpu_list in idle_gpu_sets
            for gids in combinations(gpu_list, gpus_per_node)
            if sum(set(gids).issubset(gpus) for gpus in idle_gpu_sets) >= num_nodes
        )
        gpu_ids = next(common_gpu_ids, None)

        
        if gpu_ids is None:
            raise InsufficientResources("Not enough nodes have a matching set of idle GPU IDs")

        nodes = [
            node
            for node, gpu_set in idle_gpu_map.items()
            if gpu_ids.issubset(gpu_set)
        ][:num_nodes]

        for node in nodes:
            for id in gpu_ids:
                node.idle_gpus.remove(id)
                node.busy_gpus.add(id)

        logger.debug(f"Assigned GPUs {gpu_ids} on nodes: {nodes}")
        return (nodes, gpu_ids)


class MPIRunTemplate:
    @staticmethod
    def _host_str(nodes):
        node_str = ",".join(str(node) for node in nodes)
        return f"--host {node_str}"

    @staticmethod
    def _env_str(envs):
        envstrs = (f'-x {var}="{val}"' for var, val in envs.items())
        return " ".join(envstrs)

    @staticmethod
    def render(
        command_line: str,
        nodes: List[ComputeNode],
        num_ranks: int,
        ranks_per_node: int, 
        gpu_ids=None,
        envs_dict=None,
        hostfile=None,
        
    ):
        if hostfile is None:
            hostfile = os.environ["COBALT_NODEFILE"]

        hosts = MPIRunTemplate._host_str(nodes)

        if envs_dict is None:
            envs_dict = {}
        envs_dict["MEDULLA_IDENTITY_FILE"] = IDENTITY_FILE
        if gpu_ids:
            envs_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(id) for id in gpu_ids)
        envs = MPIRunTemplate._env_str(envs_dict)

        return (
            f"mpirun -hostfile {hostfile} --oversubscribe --bind-to none "
            f"-n {num_ranks} -npernode {ranks_per_node} "
            f"{envs} {hosts} {command_line}"
        )




class MPIRun:
    ENVIRON_SETUP = []

    @staticmethod
    def set_preamble_commands(*cmds):
        MPIRun.ENVIRON_SETUP = list(cmds)

    def __init__(
        self,
        cmd_line: str,
        node_list: List[ComputeNode],
        ranks_per_node: int,
        gpu_ids: Set[int],
        output_file,
        cwd=None,
        envs_dict: Dict[str, str] = None,
    ):
        self.nodes = node_list
        self.gpu_ids = gpu_ids
        self.outfile = open(output_file, 'wb') if isinstance(output_file, str) else output_file

        mpi_command = MPIRunTemplate.render(
            command_line=cmd_line,
            nodes=node_list,
            num_ranks=len(node_list) * ranks_per_node,
            ranks_per_node=ranks_per_node,
            gpu_ids=gpu_ids,
            envs_dict=envs_dict,
        )

        args = ' && '.join(MPIRun.ENVIRON_SETUP + [mpi_command])
        logger.info(f"Popen: {args}")
        self.process = subprocess.Popen(
            args=args,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            stdout=self.outfile,
            stderr=subprocess.STDOUT
        )


    def free_nodes(self):
        for node in self.nodes:
            node.free_gpus(self.gpu_ids)

    def poll(self):
        retcode = self.process.poll()
        if retcode is not None:
            self.outfile.close()
            self.free_nodes()
        return retcode

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        fp.write(",".join(str(i) for i in range(128)))
        fp.flush()
        node_manager = ComputeNodeManager(hostfile=fp.name)

    nodes, gpus = node_manager.request(num_nodes=128, gpus_per_node=1)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=6)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=7)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
    print(nodes, gpus)
    nodes, gpus = node_manager.request(num_nodes=4, gpus_per_node=1)
    print(nodes, gpus)
    try:
        nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)
    except InsufficientResources:
        print("There were not enough resources!")
