import os
import glob
from typing import Tuple
import h5py
import numpy as np 
import tensorflow as tf
import MDAnalysis as mda

from operator import mul
from functools import reduce bvc  
from MDAnalysis.analysis import distances
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .model_tf2 import CVAE
from ddmd.utils import logger
from ddmd.utils import get_numoflines
from ddmd.utils import yml_base, BaseSettings
from ddmd.utils import create_path, get_dir_base


class ml_runs(yml_base): 
    """
    Run ML training 

    Parameters
    ----------
    pdb_file : ``str``
        Coordinate file, can also use topology file

    md_path : ``str`` 
        Path of MD simulations, where all the simulation information
        is stored

    n_train_start : ``int`` 
        Number of frame to start training 
    """
    def __init__(self, 
        pdb_file, 
        md_path,
        n_train_start=50, 
        ) -> None:
        super().__init__()
        self.pdb_file = pdb_file
        self.md_path = md_path
        self.n_train_start = n_train_start

    def get_numberofFrames(self): 
        '''
        This method assumes log and traj are with the same ouput 
        frequency
        '''
        log_files = sorted(glob.glob(f"{self.md_path}/md_run_*/*.log"))
        return sum(get_numoflines(log)-1 for log in log_files)

    def get_contact_maps(self, 
            atom_sel='name CA', 
            cutoff=8, 
            ): 
        dcd_files = sorted(glob.glob(f"{self.md_path}/md_run_*/*.dcd"))
        mda_u = mda.Universe(self.pdb_file, dcd_files)
        ca = mda_u.select_atoms(atom_sel)
        
        cm_list = []
        for _ in mda_u.trajectory: 
            cm = (distances.self_distance_array(ca.positions) < cutoff) * 1.0
            cm_list.append(cm)

        return np.array(cm_list)

    def get_train_data(self, strides): 
        contact_maps = self.get_contact_maps()
        padding = reduce(mul, [i[0] for i in strides])
        padding = max(2, padding)
        logger.debug(f"  Padding {padding} on the contact maps...")
        cvae_input = cm_to_cvae(contact_maps, padding=padding)
        logger.debug(f"cvae input shape: {cvae_input.shape}")
        train_data, val_data = data_split(cvae_input)

        return train_data, val_data   

    def build_vae(self, 
            latent_dim=3, 
            n_conv_layers=4, 
            feature_maps=[16, 16, 16, 16], 
            filter_shapes=[(3, 3), (3, 3), (3, 3), (3, 3)], 
            strides=[(1, 1), (1, 1), (1, 1), (1, 1)], 
            dense_layers=1, 
            dense_neurons=[128], 
            dense_dropouts=[0], 
            **kwargs
            ):
        train_data, val_data = self.get_train_data(strides)

        image_size = train_data.shape[1:-1]
        channel = train_data.shape[-1]
        cvae = CVAE(
                image_size, channel, 
                n_conv_layers, feature_maps,
                filter_shapes, strides, 
                dense_layers, dense_neurons,
                dense_dropouts, latent_dim, 
                **kwargs)
        return cvae, train_data, val_data

    def train_cvae(self, 
            batch_size=256, 
            epochs=100,
            **kwargs): 

        cvae, train_data, val_data = self.build_vae(**kwargs)
        cvae.train(train_data, batch_size, epochs=epochs, 
                    validation_data=val_data)
        return cvae

    def ddmd_run(self, retrain_freq=1.5, **kwargs): 
        retrain_lvl = 0
        while True: 
            # decide whether to start training
            n_frames = self.get_numberofFrames()
            if n_frames < self.n_train_start: 
                # logger.debug(f" Collected {n_frames} out of "\
                #         f"{self.n_train_start} frames for training")
                continue
            else: 
                self.n_train_start = n_frames * retrain_freq
                retrain_lvl += 1
                logger.debug(f"Starting training with {n_frames} frames...")
            
            cvae = self.train_cvae(**kwargs)
            save_path = create_path(sys_label=f'retrain_{retrain_lvl}', 
                                dir_type='vae')
            cvae.save(f"{save_path}/cvae_weight.h5")


def cm_to_cvae(cm_data, padding=2): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm) for cm in cm_data])

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


def data_split(x, train_size=.7, test_size=False): 
    train, val = train_test_split(x, train_size=train_size)
    if test_size: 
        val, test_data = train_test_split(val, train_size=test_size)
        return train, val, test_data
    else: 
        return train, val