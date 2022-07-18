import os
import glob
import json
from typing import List
import numpy as np
import pandas as pd
import MDAnalysis as mda
from sklearn.neighbors import LocalOutlierFactor 

from ddmd.ml import ml_base
from ddmd.utils import logger, yml_base

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

    def build_md_df(self, ref_pdb=None, **kwargs): 
        dcd_files = self.get_md_runs(form='all')
        df_entry = []
        for dcd in dcd_files: 
            try: 
                mda_u = mda.Universe(self.pdb_file, dcd)
            except: 
                logger.debug(f"Skipping {dcd}...")
                continue

            for ts in mda_u.trajectory: 
                local_entry = {'pdb': self.pdb_file, 
                            'dcd': dcd, 
                            'frame': ts.frame}
                df_entry.append(local_entry)
        if ref_pdb: 
            pass
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

    def ddmd_run(self, n_outlier=50, **kwargs): 
        iteration = 0
        while True: 
            trained_models = self.get_trained_models() 
            if trained_models == []: 
                continue 
            df = self.build_md_df(**kwargs)
            # df = df.sort_values('lof_score').head(n_outlier)

        

def lof_score_from_embeddings(
            embeddings, n_neighbors=20, n_jobs=None, **kwargs):
    clf = LocalOutlierFactor(
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs,**kwargs).fit(embeddings) 
    return clf.negative_outlier_factor_


if __name__ == "__main__": 
    pdb_file = '/lambda_stor/homes/heng.ma/Research/ddmd/ddmd/data/pdbs/bba/1FME-unfolded.pdb'
    md_path = '../../examples'
    vae_path = '../../examples'
    runs = inference_run(pdb_file, md_path, vae_path)