import os 
import h5py
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
    cm_filepath = os.path.abspath(os.path.join('.', 'omm_runs*/*_cm.h5'))

# delete previously existing cvae input file 
if os.path.exists('cvae_input.h5'):
	os.remove('cvae_input.h5')

num_frame_cap = int(args.l)

while not os.path.exists("halt"): 
	cm_files = sorted(glob(cm_filepath)) 

	if cm_files == []: 
	    raise IOError("No h5 file found, recheck your input filepath") 

	# Get number of frames that simulation generates 
	n_frames = get_num_frames(cm_files)

	if n_frames > num_frame_cap:
		# Compress all .h5 files into one in cvae format 
		cvae_input = get_cvae_input(cm_files)

		# Create .h5 as cvae input
		cvae_input_file = 'cvae_input.h5'
		cvae_input_save = h5py.File(cvae_input_file, 'w')
		cvae_input_save.create_dataset('contact_maps', data=cvae_input)
		cvae_input_save.close()

		num_frame_cap = int(num_frame_cap * 1.5) 
		print(f"Update frame cap to {num_frame_cap}")
