from itertools import combinations
from typing import List, Dict, Tuple, Set
import os
import getpass
import GPUtil
import subprocess
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


class InsufficientResources(Exception): pass


class GPUManager:
    def __init__(self, maxLoad=.1, maxMemory=.1):
        self.maxLoad = maxLoad
        self.maxMemory = maxMemory

    def request(self, num_gpus:int) -> Set[int]:
        gpu_ids = GPUtil.getAvailable(
                limit=num_gpus, 
                maxLoad = self.maxLoad, 
                maxMemory = self.maxMemory)
        return gpu_ids


class MPIRunTemplate:
    @staticmethod
    def _env_str(envs):
        envstrs = (f'-x {var}="{val}"' for var, val in envs.items())
        return " ".join(envstrs)

    @staticmethod
    def render(
        command_line: str,
        num_ranks: int,
        gpu_ids=None,
        envs_dict=None,
    ):

        if envs_dict is None:
            envs_dict = {}
        if gpu_ids:
            envs_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(id) for id in gpu_ids)
        envs = MPIRunTemplate._env_str(envs_dict)

        return (
            f"mpirun -n {num_ranks} "
            f"{envs} {command_line}"
        )


class MPIRun:
    ENVIRON_SETUP = []

    @staticmethod
    def set_preamble_commands(*cmds):
        MPIRun.ENVIRON_SETUP = list(cmds)

    def __init__(
        self,
        cmd_line: str,
        num_ranks: int,
        output_file,
        gpu_ids=None,
        cwd=None,
        envs_dict: Dict[str, str] = None,
    ):
        self.gpu_ids = gpu_ids
        self.outfile = open(output_file, 'wb') if isinstance(output_file, str) else output_file

        mpi_command = MPIRunTemplate.render(
            command_line=cmd_line,
            num_ranks=num_ranks,
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

    def poll(self):
        retcode = self.process.poll()
        if retcode is not None:
            self.outfile.close()
        return retcode

    def kill(self): 
        self.process.kill()

if __name__ == "__main__":
    gpu_manager = GPUManager()

    gpus = gpu_manager.request(num_gpus=1)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=2)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=4)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=4)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=5)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=6)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=7)
    print(gpus)
