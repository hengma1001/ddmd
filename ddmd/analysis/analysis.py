import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from ddmd.inference import inference_run
from ddmd.utils import get_dir_base


class analysis_run(inference_run): 
    def __init__(self, pdb_file, md_path, ml_path) -> None:
        super().__init__(pdb_file, md_path, ml_path)
    
    def run(self, n_clusters=500, random_state=42, init='k-means++', **kwargs): 
        # base analysis from infer
        df = self.build_md_df(form='done', **kwargs)
        df['sys_label'] = [get_dir_base(i) for i in df['dcd']]
        df['gpu_id'] = [i.split('_')[2] for i in df['sys_label']]
        embeddings = np.array(df['embeddings'].to_list())

        # kmeans cluster
        df['cluster_label'] = kmeans_emb_clustering(
                embeddings, n_clusters=n_clusters, 
                init=init, random_state=random_state)
        # sort the dataframe to represent time evo
        df.sort_values(by=['gpu_id', 'sys_label', 'frame'], inplace=True)
        # save df 
        df.to_pickle("result.pkl")

        
        # reshape dtraj
        dtrajs = {}
        for gpu_id in df.gpu_id.unique(): 
            dtraj = df[df.gpu_id == gpu_id].cluster_label.to_numpy()
            dtrajs[f'gpu_{gpu_id}'] = dtraj
        
        dtrajs['total'] = traj_stack(dtrajs.values())

        sampling_effs = []
        for run in dtrajs: 
            sampling_effs.append(
                {'run': run, 'sampling_eff': get_sampling(dtrajs[run])}
            )
        sampling_effs = pd.DataFrame(sampling_effs)
        sampling_effs.to_pickle('sampling.pkl')


def kmeans_emb_clustering(embeddings, n_clusters=500, random_state=42, init='k-means++'): 
    kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            init=init, random_state=random_state
        ).fit(embeddings)
    return kmeans.labels_


def traj_stack(dtrajs): 
    y_len = max(len(i) for i in dtrajs)
    dtraj_stacks = np.ones([len(dtrajs), y_len]) * -1
    for i, dtraj in enumerate(dtrajs):
        dtraj_stacks[i, :len(dtraj)] = dtraj
    return dtraj_stacks[dtraj_stacks != -1]

def get_sampling(traj): 
    sampled = []
    n_sampled = []
    for i in tqdm(traj): 
        if i not in sampled: 
            sampled.append(i)
        n_sampled.append(len(sampled))
    return n_sampled