import os
import sys
import errno
import argparse 
from cvae.CVAE import run_cvae  
import numpy as np 


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--h5_file", dest="f", 
    default='cvae_input.h5', 
    help="Input: contact map h5 file")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument(
    "-d", "--dim", default=3, 
    help="Number of dimensions in latent space")
parser.add_argument("-gpu", default=0, help="gpu_id")

args = parser.parse_args()

cvae_input = args.f
hyper_dim = int(args.dim) 
gpu_id = args.gpu
work_dir = os.getcwd() 

old_num_frame = 0 
while not os.path.exists("halt"):
    if not os.path.exists(cvae_input):
        continue 
    else: 
        cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
        cm_data = cm_h5['contact_maps']
        num_frame = cm_data.shape[0]
        if num_frame == old_num_frame:
            continue 
        elif num_frame > old_num_frame:
            # run cvae 
            cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim)

            time_label = int(time.time())
            cvae_path = f'cvae_runs{hyper_dim:02}_{time_label}'
            os.makedir(cvae_path)

            model_weight = cvae_path + '/cvae_weight.h5' 
            model_file = cvae_path + '/cvae_model.h5' 
            loss_file = cvae_path + '/loss.npy' 

            cvae.model.save_weights(model_weight)
            cvae.save(model_file)
            np.save(loss_file, cvae.history.losses) 
        else: 
            raise Exception("New frame number is smaller than the old. ")
