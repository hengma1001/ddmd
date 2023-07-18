import inspect
import os 
import time
import yaml
import logging
import argparse
import MDAnalysis as mda

from typing import Union
from pathlib import Path
from typing import Type, TypeVar
# from pydantic import BaseSettings as _BaseSettings

PathLike = Union[str, Path]
_T = TypeVar("_T")


def build_logger(debug=0):
    logger_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


def dict_from_yaml(yml_file): 
    return yaml.safe_load(open(yml_file, 'r'))

def dict_to_yaml(dict_t, yml_file): 
    with open(yml_file, 'w') as fp: 
        yaml.dump(dict_t, fp, default_flow_style=False)

# class BaseSettings(_BaseSettings):
#     def dump_yaml(self, cfg_path: PathLike) -> None:
#         with open(cfg_path, mode="w") as fp:
#             yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

#     @classmethod
#     def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
#         with open(filename) as fp:
#             raw_data = yaml.safe_load(fp)
#         return cls(**raw_data)  # type: ignore[call-arg]


class yml_base(object): 
    def dump_yaml(self, cfg_path: PathLike) -> None: 
        dict_to_yaml(self.get_setup(), cfg_path)


def create_path(dir_type='md', sys_label=None, time_stamp=True): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    time_label = int(time.time())
    dir_path = f'{dir_type}_run'
    if sys_label: 
        dir_path = f'{dir_path}_{sys_label}'
    if time_stamp: 
         time_path = f'{dir_path}_{time_label}'
         while True:
            try: 
                os.makedirs(time_path)
                dir_path = time_path
                break
            except: 
                time_path = f'{dir_path}_{time_label + 1}'

    else: 
        os.makedirs(dir_path, exist_ok=True)
    return os.path.abspath(dir_path)


def get_dir_base(file_path): 
    return os.path.basename(os.path.dirname(file_path))


def touch_file(file): 
    """
    create an empty file for bookkeeping sake
    """
    with open(file, 'w'): 
        pass


def get_numoflines(file): 
    return sum(1 for _ in open(file, 'r'))


def get_function_kwargs(func): 
    sig = inspect.signature(func)
    return [i for i in sig.parameters]

def separate_kwargs(func, kwargs): 
    input_kwargs = {}
    input_keys = get_function_kwargs(func)
    for key in input_keys: 
        if key in kwargs: 
            input_kwargs[key] = kwargs.pop(key)
    return input_kwargs, kwargs

def write_pdb_frame(pdb, dcd, frame:int, save_path=None): 
    mda_u = mda.Universe(pdb, dcd)
    mda_u.trajectory[frame]
    pdb_save_name = f"{get_dir_base(dcd)}_{frame:06}.pdb"
    if save_path: 
        pdb_save_name = f"{save_path}/{pdb_save_name}"
    mda_u.atoms.write(pdb_save_name)
    return os.path.abspath(pdb_save_name)


def backup_path(filepath): 
    while os.path.exists(filepath): 
        pass


def ddmd_abspath(filepath): 
    if filepath: 
        return os.path.abspath(filepath)
    else: 
        return None
    

def missing_hydrogen(pdb_file):
    """
    Check whether a pdb file contains H atoms

    Parameters
    ----------
    pdb_file : str
        path to input pdb file

    Returns
    -------
    missingH : bool
        True if missing H, false otherwise
    """
    mda_u = mda.Universe(pdb_file)
    hydrogens = mda_u.select_atoms('name H*')
    missingH = True if hydrogens.n_atoms == 0 else False
    return missingH