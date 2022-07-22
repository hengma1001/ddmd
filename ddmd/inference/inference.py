import os
import glob
import json
import time
from typing import List
import numpy as np
import pandas as pd
import MDAnalysis as mda

from MDAnalysis.analysis import rms
from sklearn.neighbors import LocalOutlierFactor 

from ddmd.ml import ml_base
from ddmd.utils import build_logger, create_path, get_numoflines, get_dir_base

logger = build_logger()

class inference_run(ml_base): 
    """
    Inferencing between MD and ML

    Parameters
    ----------
    pdb_file : ``str``
        Coordinate file, can also use topology file

    md_path : ``str`` 
        Path of MD simulations, where all the simulation information
        is stored

    ml_path : ``str``
        Path of VAE or other ML traning directory, used to search 
        trained models
    """
    def __init__(self, 
        pdb_file, 
        md_path,
        ml_path,
        ) -> None:
        super().__init__(pdb_file, md_path)
        self.ml_path = ml_path
        self.vae = None
        self.outlier_path = create_path(dir_type='inference')

    def get_trained_models(self): 
        return sorted(glob.glob(f"{self.ml_path}/vae_run_*/*h5"))

    def get_md_runs(self, form : str='all') -> List: 
        if form.lower() == 'all': 
            return sorted(glob.glob(f'{self.md_path}/md_run*/*dcd'))
        elif form.lower() == 'done': 
            md_done = sorted(glob.glob(f'{self.md_path}/md_run*/Done'))
            return [f'{os.path.dirname(i)}/output.dcd' for i in md_done]
        elif form.lower() == 'running': 
            return [i for i in self.get_md_runs(form='all') if i not in self.get_md_runs(form='done')]
        else: 
            raise("Form not defined, using all, done or running ...")

    def build_md_df(self, ref_pdb=None, atom_sel="name CA", **kwargs): 
        dcd_files = self.get_md_runs(form='all')
        df_entry = []
        if ref_pdb: 
            ref_u = mda.Universe(ref_pdb)
            sel_ref = ref_u.select_atoms(atom_sel)
        for dcd in dcd_files: 
            try: 
                mda_u = mda.Universe(self.pdb_file, dcd)
                sel_atoms = mda_u.select_atoms(atom_sel)
            except: 
                logger.info(f"Skipping {dcd}...")
                continue

            for ts in mda_u.trajectory: 
                local_entry = {'pdb': self.pdb_file, 
                            'dcd': os.path.abspath(dcd), 
                            'frame': ts.frame}
                if ref_pdb: 
                    rmsd = rms.rmsd(
                            sel_atoms.positions, sel_ref.positions, 
                            superposition=True)
                    local_entry['rmsd'] = rmsd
                # possible new analysis
                df_entry.append(local_entry)
        
        df = pd.DataFrame(df_entry)
        embeddings = self.get_embeddings()
        df['embeddings'] = embeddings.tolist()
        outlier_score = lof_score_from_embeddings(embeddings, **kwargs)
        df['lof_score'] = outlier_score
        return df

    def get_embeddings(self): 
        """
        getting the last model so far
        """
        # get weight 
        vae_weight = self.get_trained_models()[-1]
        vae_setup = os.path.join(os.path.dirname(vae_weight), 'cvae.json')
        vae_label = os.path.basename(os.path.dirname(vae_weight))
        # get conf 
        vae_config = json.load(open(vae_setup, 'r'))
        if self.vae is None: 
            self.vae, cvae_input = self.build_vae(**vae_config)
        else: 
            cvae_input = self.get_vae_input(**vae_config)
        # load weight
        logger.info(f" ML nn created and loaded weight from {vae_label}")
        self.vae.load(vae_weight)
        embeddings = self.vae.return_embeddings(cvae_input)
        return embeddings

    def ddmd_run(self, n_outlier=50, md_threshold=0.75, **kwargs): 
        iteration = 0
        while True: 
            trained_models = self.get_trained_models() 
            if trained_models == []: 
                continue
            md_done = self.get_md_runs(form='done')
            if md_done == []: 
                continue
            else: 
                len_md_done = \
                    get_numoflines(md_done[0].replace('dcd', 'log')) - 1
            # build the dataframe and rank outliers 
            df = self.build_md_df(**kwargs)
            df_outliers = df.sort_values('lof_score').head(n_outlier)
            if 'ref_pdb' in kwargs: 
                df_outliers = df_outliers.sort_values('rmsd')
            # assess simulations 
            sim_running = self.get_md_runs(form='running')
            sim_to_stop = [i for i in sim_running \
                    if i not in set(df_outliers['dcd'].to_list())]
            # only stop simulations that have been running for a while
            # 3/4 done
            sim_to_stop = [i for i in sim_to_stop \
                    if get_numoflines(i.replace('dcd', 'log')) \
                    > len_md_done * md_threshold]
            for i, sim in enumerate(sim_to_stop): 
                sim_path = os.path.dirname(sim)
                outlier = df_outliers.iloc[i]
                outlier.to_json(f"{sim_path}/new_pdb")
                logger.info(f"Writing new pdb from frame "\
                    f"{outlier['frame']} of {get_dir_base(outlier['dcd'])} "\
                    f"to {get_dir_base(sim)}")
                # write_pdb_frame(, dcd, frame, save_path=None)

            logger.info(f"\n=======> Done iteration {iteration} <========\n")
            time.sleep(1)
            iteration += 1
        

def lof_score_from_embeddings(
            embeddings, n_neighbors=20, n_jobs=None, **kwargs):
    clf = LocalOutlierFactor(
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs,**kwargs).fit(embeddings) 
    return clf.negative_outlier_factor_



