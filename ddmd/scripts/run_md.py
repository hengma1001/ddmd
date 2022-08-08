#!env python

from ddmd.sim import omm_sim
from ddmd.utils import dict_from_yaml, parse_args


def main(args): 
    sim_setup = dict_from_yaml(args.config)
    # set max iter to 1B if unspecified
    max_iter = sim_setup.pop('max_iter') if 'max_iter' in sim_setup else 1e9

    sim_imp = omm_sim(**sim_setup)
    sim_imp.ddmd_run(iter=max_iter)

if __name__ == '__main__': 
    args = parse_args()
    main(args)
