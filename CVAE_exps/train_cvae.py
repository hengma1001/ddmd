import os
import gc
import sys
import h5py
import time
import logging
import argparse 
import numpy as np 
from keras import backend as K

from cvae.CVAE import CVAE, data_split 

debug = 0
logger_level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--h5_file", dest="f", 
    default='cvae_input.h5', 
    help="Input: contact map h5 file")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument(
    "-d", "--dim", default=3, 
    help="Number of dimensions in latent space")
parser.add_argument("-g", "--gpu", default=0, help="gpu_id")
parser.add_argument(
    "-b", "--batch", default=1000, 
    help="Batch size for CVAE training") 

args = parser.parse_args()
cvae_input = args.f
hyper_dim = int(args.dim) 
gpu_id = args.gpu
batch_size = int(args.batch)
work_dir = os.getcwd() 

# setting up default parameters
skip = 1 
epochs = 100

# setting up gpu info 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

# waiting for inputs
while not os.path.exists(cvae_input): 
    pass

# create cvae instance
time.sleep(10)
cm_h5 = h5py.File(cvae_input, 'r', libver='latest', swmr=True)
cm_data = cm_h5['contact_maps']
input_shape = cm_data.shape
cvae = CVAE(input_shape[1:], hyper_dim)
cm_h5.close()

# training until aborting 
old_num_frame = 0
while not os.path.exists("../halt"): 
    # getting input len
    try: 
        cm_h5 = h5py.File(cvae_input, 'r', libver='latest', swmr=True)
    except: 
        continue
    cm_data = cm_h5['contact_maps']
    num_frame = cm_data.shape[0]
    # skipping 
    if num_frame == old_num_frame:
        cm_h5.close()
        logger.debug('No new inputs, skipping')
        continue 
    # training 
    elif num_frame > old_num_frame:
        logger.debug(f'Retraining with {num_frame - old_num_frame} new frames...')
        old_num_frame = num_frame
        time_label = int(time.time())
        cvae_path = f'cvae_runs_{hyper_dim:03}_{time_label}'
        os.makedirs(cvae_path, exist_ok=True)

        # split the data 
        cm_train, cm_val = data_split(cm_data.value) 
        # run cvae 
        cvae.train(
                cm_train, 
                validation_data=cm_val, 
                batch_size = batch_size, 
                epochs=epochs)
        # cvae = run_cvae(
        #         gpu_id, cvae_input, 
        #         hyper_dim=hyper_dim, 
        #         batch_size=batch_size)
        
        model_weight = cvae_path + '/cvae_weight.h5' 
        model_file = cvae_path + '/cvae_model.h5' 
        loss_file = cvae_path + '/loss.npy' 

        cvae.model.save_weights(model_weight)
        cvae.save(model_file)
        np.save(loss_file, cvae.history.losses) 

        # del cvae 
        # gc.collect() 
        # K.clear_session()
        print(f"Finished training with {num_frame} conformers...") 
    else: 
        raise Exception("New frame number is smaller than the old. ")
