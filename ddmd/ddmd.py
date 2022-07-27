import os
import time
import ddmd
import shutil
from ddmd.task import Run, GPUManager
from ddmd.utils import build_logger, create_path
from ddmd.utils import dict_from_yaml, dict_to_yaml

exe_scripts = {
        'md': 'run_md.py', 'ml': 'run_ml.py', 
        'infer': 'run_infer.py'}

logger = build_logger()

class ddmd_run(object): 
    """
    Set up the ddmd run 

    Parameters
    ----------
    cfg_yml : ``str``
        Main yaml file for setting up runs 
    
    md_only : ``bool``
        Whether to run ml and infer
    """
    def __init__(self, cfg_yml) -> None:
        self.cfg_yml = cfg_yml
        self.ddmd_setup = dict_from_yaml(self.cfg_yml)
        if 'conda_env' in self.ddmd_setup: 
            conda_path = self.ddmd_setup['conda_env']
            self.python_exe = f"{conda_path}/bin/python" 
        else: 
            self.python_exe = 'python'
        self.md_only = self.ddmd_setup['md_only'] if 'md_only' in self.ddmd_setup else False
        work_dir = self.ddmd_setup['output_dir']
        cont_run =self.ddmd_setup['continue'] if 'continue' in self.ddmd_setup else False
        if os.path.exists(work_dir) and not cont_run: 
            bkup_dir = work_dir + f'_{int(time.time())}'
            shutil.move(work_dir, bkup_dir)
            logger.info(f"Back up old {work_dir} to {bkup_dir}")
        os.makedirs(work_dir)
        os.chdir(work_dir)

        # logging 
        self.log_dir = 'run_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # manage GPUs
        self.n_sims=self.ddmd_setup['n_sims']
        if self.md_only: 
            n_gpus = self.n_sims 
            logger.info(f"Running only {self.n_sims} simulations...")
        else:
            n_gpus = self.n_sims + 2
        self.gpu_ids = GPUManager().request(num_gpus=n_gpus)
        logger.info(f"Available {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        # if not enough GPUs, reconf the workflow
        if len(self.gpu_ids) < n_gpus: 
            self.n_sims = self.n_sims + len(self.gpu_ids) - n_gpus
            n_gpus = len(self.gpu_ids)
            logger.info("Not enough GPUs avaible for all the runs, and "\
                f"reduce number of MD runs to {self.n_sims}")
            logger.info(f"New configuration: {self.n_sims} simulations, "\
                f"1 training, and 1 inference node. Need {n_gpus} GPUs. ")

    def build_tasks(self): 
        md_setup = self.ddmd_setup['md_setup']
        self.md_path = create_path(dir_type='md', time_stamp=False)
        md_yml = f"{self.md_path}/md.yml"
        dict_to_yaml(md_setup, md_yml)
        if self.md_only: 
            self.ml_path = None
            self.infer_path = None
            return md_yml, None, None

        ml_setup = self.ddmd_setup['ml_setup'].copy() 
        ml_setup['pdb_file'] = md_setup['pdb_file']
        ml_setup['md_path'] = self.md_path
        self.ml_path = create_path(dir_type='ml', time_stamp=False)
        ml_yml = f"{self.ml_path}/ml.yml"
        dict_to_yaml(ml_setup, ml_yml)

        infer_setup = self.ddmd_setup['infer_setup']
        infer_setup['pdb_file'] = md_setup['pdb_file']
        infer_setup['md_path'] = self.md_path
        infer_setup['ml_path'] = self.ml_path
        self.infer_path = create_path(dir_type='infer', time_stamp=False)
        infer_yml = f"{self.infer_path}/infer.yml"
        dict_to_yaml(infer_setup, infer_yml)
        return md_yml, ml_yml, infer_yml

    def submit_job(self, 
            yml_file, work_path,
            n_gpus=1, job_type='md', 
            type_ind=-1): 
        exe_py = f"{ddmd.__path__[0]}/scripts/{exe_scripts[job_type]}"
        run_cmd = f"{self.python_exe} {exe_py} -c {yml_file}"
        # setting up output log file
        output_file = f"./{self.log_dir}/{job_type}"
        if type_ind >= 0: 
            output_file = f"{output_file}_{type_ind}"
        # get gpu ids for current job 
        gpu_ids = [self.gpu_ids.pop() for _ in range(n_gpus)]
        run = Run(
            cmd_line=run_cmd,
            gpu_ids=gpu_ids,
            output_file=output_file,
            cwd=work_path, # can be a different working directory
            envs_dict=None, # can be a dictionary of environ vars to add
        )
        return run
        
    def run(self): 
        "create and submit ddmd jobs "
        md_yml, ml_yml, infer_yml = self.build_tasks()
        runs = []
        # md
        for i in range(self.n_sims): 
            md_run = self.submit_job(md_yml, self.md_path, n_gpus=1, 
                    job_type='md', type_ind=i)
            runs.append(md_run)
            # avoid racing during dir creation
            # time.sleep(1)
        if self.md_only: 
            return runs
        # ml 
        ml_run = self.submit_job(ml_yml, self.ml_path, n_gpus=1, 
                job_type='ml')
        runs.append(ml_run)
        # infer
        infer_run = self.submit_job(infer_yml, self.infer_path, 
                n_gpus=1, job_type='infer')
        runs.append(infer_run)
        return runs 