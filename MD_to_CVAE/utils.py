import h5py 
import datetime
import numpy as np


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


def get_num_frames(cm_files): 
    num_frams = 0 
    for cm_file in cm_files: 
        with h5py.File(cm_file, 'r', libver='latest', swmr=True) as cm_h5:
            num_frams += cm_h5['contact_maps'].shape[1] 
    return num_frams


def cm_to_cvae(cm_data, padding=2): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm) for cm in cm_data.T])

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def get_cvae_input(cm_list, padding=2): 
    cvae_input_all = [] 
    for cm_file in cm_list: 
        cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
        cm_data = cm_h5['contact_maps']
        cvae_input = cm_to_cvae(np.array(cm_data), padding=padding)
        cm_h5.close()
        cvae_input_all.append(cvae_input)
    cvae_input_all = np.concatenate(cvae_input_all, axis=0)
    return cvae_input_all


def stamp_to_time(stamp): 
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
