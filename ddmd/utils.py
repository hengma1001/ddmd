import os 
import yaml
import json
from typing import Union
from pathlib import Path
from pydantic import BaseSettings as _BaseSettings

PathLike = Union[str, Path]


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore[call-arg]


def create_md_path(label, sys_label=None): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    if sys_label: 
        md_path = f'md_run_{sys_label}_{label}'
    else: 
         md_path = f'md_run_{label}'
    try:
        os.mkdir(md_path)
        return md_path
    except: 
        return create_md_path(label + 1, sys_label=sys_label)


def get_dir_base(file_path): 
    return os.path.basename(os.path.dirname(file_path))

def touch_file(file): 
    """
    create an empty file for bookkeeping sake
    """
    with open(file, 'w'): 
        pass
