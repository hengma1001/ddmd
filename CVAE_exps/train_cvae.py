import os
import gc
import sys
import h5py
import time
import errno
import argparse 
import numpy as np 
from keras import backend as K

from cvae.CVAE import run_cvae  


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
parser.add_argument(
    "-b", "--batch_size", default=1000, 
    help="Batch size for CVAE training") 

args = parser.parse_args()

cvae_input = args.f
hyper_dim = int(args.dim) 
gpu_id = args.gpu
work_dir = os.getcwd() 

old_num_frame = 0 
time.sleep(600)
while not os.path.exists("../halt"):
    if not os.path.exists(cvae_input):
        continue 
    else: 
        time.sleep(60)
        cm_h5 = h5py.File(cvae_input, 'r', libver='latest', swmr=True)
        cm_data = cm_h5['contact_maps']
        num_frame = cm_data.shape[0]
        if num_frame == old_num_frame:
            continue 
        elif num_frame > old_num_frame:
            old_num_frame = num_frame
            # run cvae 
            cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim)

            time_label = int(time.time())
            cvae_path = f'cvae_runs_{hyper_dim:02}_{time_label}'
            os.mkdir(cvae_path)

            model_weight = cvae_path + '/cvae_weight.h5' 
            model_file = cvae_path + '/cvae_model.h5' 
            loss_file = cvae_path + '/loss.npy' 

            cvae.model.save_weights(model_weight)
            cvae.save(model_file)
            np.save(loss_file, cvae.history.losses) 

            del cvae 
            gc.collect() 
            K.clear_session()
            print(f"Finished training with {num_frame} conformers...") 
        else: 
            raise Exception("New frame number is smaller than the old. ")
