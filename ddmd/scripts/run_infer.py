#!env python

from ddmd.inference import inference_run
from ddmd.utils import dict_from_yaml, parse_args, separate_kwargs

args = parse_args()
infer_setup = dict_from_yaml(args.config)

infer_kwargs, ddmd_kwargs = separate_kwargs(inference_run, infer_setup)
runs = inference_run(**infer_kwargs)
runs.ddmd_run(**ddmd_kwargs)
