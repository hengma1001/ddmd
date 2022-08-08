import time
import ddmd
from ddmd.ddmd import ddmd_run
from ddmd.utils import parse_args

logger = ddmd.utils.build_logger(debug=1)


args = parse_args()
cfg_file = args.config

ddmd_runs = ddmd_run(cfg_file)
runs = ddmd_runs.run() 

print("waiting on", len(runs), "runs to finish...")
try: 
    while runs:
        # Clean up runs which finished (poll() returns process code)
        runnings = [run for run in runs if run.poll() is None]
        if len(runs) != len(runnings):
            logger.info(f"waiting on {len(runnings)} runs to finish...")
            runs = runnings 
        time.sleep(5)
    logger.info("All done!")
except KeyboardInterrupt: 
    for p in runs: 
        p.kill()
    logger.info("cleaned up!") 
