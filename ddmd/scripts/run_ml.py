#!env python

from ddmd.ml import ml_run
from ddmd.utils import dict_from_yaml, parse_args, separate_kwargs

args = parse_args()
ml_setup = dict_from_yaml(args.config)

ml_kwargs, ddmd_kwargs = separate_kwargs(ml_run, ml_setup)
runs = ml_run(**ml_kwargs)
runs.ddmd_run(**ddmd_kwargs)
