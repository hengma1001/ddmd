import time
import ddmd
from ddmd.ensemble_sim import ensemble_run
from ddmd.utils import parse_args

logger = ddmd.utils.build_logger(debug=1)

def main(args):
    cfg_file = args.config

    ensemble_runs = ensemble_run(cfg_file)
    ensemble_runs.run() 

if __name__ == '__main__': 
    args = parse_args()
    main(args)
