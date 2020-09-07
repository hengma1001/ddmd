import os 
import h5py
import time
import argparse
import warnings 
import numpy as np 
from glob import glob
from utils import get_num_frames, get_cvae_input

parser = argparse.ArgumentParser()
parser.add_argument(
	"-f", "--sim_path", dest='f', 
	help="Input: OpenMM simulation path") 
parser.add_argument(
	"-l", "--train_frames", dest='l', default=10000, 
	help="Number of frames for 1st training session, 1.5 times of which will be next iteration")

# Let's say I have a list of h5 file names 
args = parser.parse_args() 

if args.f: 
    cm_filepath = os.path.abspath(os.path.join(args.f, 'omm_runs*/*_cm.h5')) 
else: 
    warnings.warn("No input dirname given, using current directory...") 
    cm_filepath = os.path.abspath('./omm_runs*/*_cm.h5')

# delete previously existing cvae input file 
if os.path.exists('cvae_input.h5'):
	os.remove('cvae_input.h5')

num_frame_cap = int(args.l)

n_iter = 0 
while not os.path.exists("../halt"): 
    # get all contact map h5 files 
    cm_files = sorted(glob(cm_filepath)) 

    if cm_files == []: 
        continue     

    # Get number of frames that simulation generates 
    while True: 
        try: 
            n_frames = get_num_frames(cm_files)
            break
        except: 
            time.sleep(60) 

    if n_frames > num_frame_cap:
        # Compress all .h5 files into one in cvae format 
        cvae_input = get_cvae_input(cm_files)

        # Create .h5 as cvae input
        while True: 
            try:
                cvae_input_file = 'cvae_input.h5'
                cvae_input_save = h5py.File(cvae_input_file, 'w')
                cvae_input_save.create_dataset('contact_maps', data=cvae_input)
                cvae_input_save.close()
                break
            except: 
                continue 

        num_frame_cap = int(n_frames * 1.6) 
        print(f"Update frame cap to {num_frame_cap}")
    elif n_iter % 10 == 0: 
        print(f"accumulated {n_frames} frames...")
        time.sleep(60)
    
    n_iter += 1 
