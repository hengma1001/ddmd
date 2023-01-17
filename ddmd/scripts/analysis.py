#!env python

from ddmd.analysis.analysis import analysis_run
from ddmd.utils import dict_from_yaml, parse_args, separate_kwargs

def main(args): 
    analysis_setup = dict_from_yaml(args.config)

    analysis_kwargs, ddmd_kwargs = separate_kwargs(analysis_run, analysis_setup)
    runs = analysis_run(**analysis_kwargs)
    ddmd_kwargs, df_kwargs = separate_kwargs(runs.ddmd_run, ddmd_kwargs)
    runs.run(**df_kwargs)

if __name__ == '__main__': 
    args = parse_args()
    main(args)
