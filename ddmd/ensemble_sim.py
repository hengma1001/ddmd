import os
import glob
import time
from typing import List
from ddmd.ddmd import ddmd_run
from ddmd.utils import build_logger, create_path
from ddmd.utils import dict_from_yaml, dict_to_yaml

logger = build_logger(debug=1)

class ensemble_run(ddmd_run): 
    """
    multiple input simulation runs 
    """
    def __init__(self, cfg_yml) -> None:
        super().__init__(cfg_yml)

    def build_tasks(self):
        md_setup = self.ddmd_setup['md_setup']
        # correcting file path
        md_setup['pdb_file'] = sorted(glob.glob(md_setup['pdb_file']))
        self.md_path = create_path(dir_type='md', time_stamp=False)

        md_ymls= []
        for i, pdb_file in enumerate(md_setup['pdb_file']): 
            md_setup_copy = md_setup.copy()
            md_setup_copy['pdb_file'] = pdb_file
            md_setup_copy['top_file'] = pdb_file[:-3] + 'top'
            md_yml = f"{self.md_path}/md_{i}.yml"
            dict_to_yaml(md_setup_copy, md_yml)
            md_ymls.append(md_yml)

        return md_ymls

    def run(self):
        "create and submit ddmd jobs "
        md_ymls = self.build_tasks()
        runs = []
        # md
        for i in range(self.n_sims): 
            md_yml = md_ymls.pop()
            ind = int(os.path.basename(md_yml)[:-4].split('_')[1])
            md_run = self.submit_job(
                    md_yml, self.md_path, n_gpus=1, 
                    job_type='md', type_ind=ind)
            runs.append(md_run)
        
        try:
            # loop through all the yamls
            while md_ymls != []: 
                runs_done = [run for run in runs if run.poll() is not None]
                if len(runs_done) > 0: 
                    for run in runs_done:
                        runs.remove(run)
                        self.gpu_ids.extend(run.gpu_ids)
                        logger.info(f"Finished run on gpu {run.gpu_ids}")

                        i += 1
                        md_yml = md_ymls.pop()
                        ind = int(os.path.basename(md_yml)[:-4].split('_')[1])
                        md_run = self.submit_job(
                                md_yml, self.md_path, n_gpus=1, 
                                job_type='md', type_ind=ind)
                        runs.append(md_run)
            # waiting for runs to finish
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
        
        
