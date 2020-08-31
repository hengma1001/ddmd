import os
import time
import random
import shutil 
import argparse 
import numpy as np 
from glob import glob
import MDAnalysis as mda
from utils import predict_from_cvae, outliers_from_latent_loc 
from utils import outliers_largeset
from utils import find_frame, write_pdb_frame 
from  MDAnalysis.analysis.rms import RMSD

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
DEBUG = 1 

# Inputs 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--md", help="Input: MD simulation directory")
parser.add_argument("-c", "--cvae", help="Input: CVAE model directory")
parser.add_argument("-p", "--pdb", help="Input: pdb file") 
parser.add_argument("-r", "--ref", default=None, help="Input: Reference pdb for RMSD") 
parser.add_argument(
    "-n", "--n_out", default=500, 
    help="Input: Approx number of outliers to gather")  
parser.add_argument(
    "-t", "--timeout", default=10, 
    help="Input: time to exam outliers for MD runs, \
          in which if no ouliers emerges in the latest\
          t nanoseconds (10^-9), MD run stops. "
    )

args = parser.parse_args()

# pdb file for MDAnalysis 
pdb_file = os.path.abspath(args.pdb) 
ref_pdb_file = os.path.abspath(args.ref) 
n_outliers = int(args.n_out)
time_out = float(args.timeout)
iteration = 0 

while not os.path.exists("halt"):
    # get all omm_runs path 
    md_path = args.md
    omm_runs = sorted(glob(os.path.join(md_path, 'omm_runs_*')))

    # Find all the trained model weights 
    cvae_path = args.cvae
    cvae_runs = sorted(glob(os.path.join(cvae_path, 'cvae_runs_*'))) 

    # need to wait for training to finish 
    while cvae_runs == []: 
        omm_runs = sorted(glob(os.path.join(md_path, 'omm_runs_*')))
        cvae_runs = sorted(glob(os.path.join(cvae_path, 'cvae_runs_*')))
    # wait for a few to minutes for cvae finish writing output
    time.sleep(120)

    # identify the latest models with lowest loss 
    model_dims = [os.path.basename(cvae).split('_')[2] for cvae in cvae_runs]
    model_dims = sorted(list(set(model_dims)))
    cvae_loss = []
    for i, dim in enumerate(model_dims): 
        latest_run = sorted(glob(os.path.join(cvae_path, f'cvae_runs_{dim}_*')))[-1] 
        model_loss = np.load(os.path.join(latest_run, 'loss.npy'))[-1]
        cvae_loss.append(model_loss)

    sel_dim = model_dims[np.argmin(cvae_loss)]
    sel_model = sorted(glob(os.path.join(cvae_path, f'cvae_runs_{sel_dim}_*')))[-1]
    sel_model_weight = sel_model + '/cvae_weight.h5'

    print(("Using model {} with {}... ".format(sel_model, min(cvae_loss))))

    # Get the predicted embeddings 
    # cm_files_list = [os.path.join(omm_run, 'output_cm.h5') for omm_run in omm_runs]
    cm_files_list = sorted(glob(os.path.join(md_path, 'omm_runs_*/output_cm.h5')))
    cm_predict, train_data_length = predict_from_cvae(
            sel_model_weight, cm_files_list, 
            hyper_dim=int(sel_dim), padding=4) 

    # A record of every trajectory length
    omm_runs = [os.path.dirname(cm_file) for cm_file in cm_files_list]
    traj_dict = dict(list(zip(omm_runs, train_data_length))) 
    print(traj_dict)

    ## Unique outliers 
    print("Starting outlier searching...")
    outlier_list_ranked, _ = outliers_from_latent_loc(
            cm_predict, n_outliers=n_outliers, n_jobs=12)  
    print("Done outlier searching...")
    # print(outlier_list_ranked)

    # Write the outliers using MDAnalysis 
    outliers_pdb_path = os.path.abspath('./outlier_pdbs') 
    os.makedirs(outliers_pdb_path, exist_ok=True) 
    print('Writing outliers in %s' % outliers_pdb_path)  

    # identify new outliers 
    new_outliers_list = [] 
    for outlier in outlier_list_ranked:
        # find the location of outlier 
        traj_dir, num_frame = find_frame(traj_dict, outlier)
        traj_file = os.path.join(traj_dir, 'output.dcd')
        # get the outlier name - traj_label + frame number 
        run_name = os.path.basename(traj_dir)
        pdb_name = f"{run_name}_{num_frame:06}.pdb"
        outlier_pdb_file = os.path.join(outliers_pdb_path, pdb_name) 

        new_outliers_list.append(outlier_pdb_file)
        # Only write new pdbs to reduce I/O redundancy. 
        if not os.path.exists(outlier_pdb_file): 
            print(f'New outlier at frame {num_frame} of {run_name}')
            outlier_pdb = write_pdb_frame(
                    traj_file, pdb_file, num_frame, outlier_pdb_file)  
         

    # Clean up outdated outliers (just for bookkeeping)
    outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb')) 
    for outlier in outliers_list: 
        if outlier not in new_outliers_list: 
            outlier_label = os.path.basename(outlier)
            print(f'Old outlier {outlier_label} is now connected to \
                a cluster and removing it from the outlier list ') 
            # os.rename(outlier, os.path.join(outliers_pdb_path, '-'+outlier_label)) 


    # Set up input configurations for next batch of MD simulations 
    ### Get the pdbs used once already 
    used_pdbs = glob(os.path.join(md_path, 'omm_runs_*/omm_runs_*.pdb'))
    used_pdbs_labels = [os.path.basename(used_pdb) for used_pdb in used_pdbs ]
    print(used_pdbs_labels) 
    ### Exclude the used pdbs 
    # outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb'))
    restart_pdbs = [outlier for outlier in new_outliers_list \
            if os.path.basename(outlier) not in used_pdbs_labels] 
    print("restart pdbs: ", restart_pdbs)

    # rank the restart_pdbs according to their RMSD to local state 
    if ref_pdb_file: 
        ref_traj = mda.Universe(ref_pdb_file) 
        outlier_traj = mda.Universe(restart_pdbs[0], restart_pdbs) 
        R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
        R.run()    
        # Make a dict contains outliers and their RMSD
        # outlier_pdb_RMSD = dict(zip(restart_pdbs, R.rmsd[:,2]))
        restart_pdbs = [pdb for _, pdb in sorted(zip(R.rmsd[:,2], restart_pdbs))] 
        if np.min(R.rmsd[:,2]) < 0.1: 
            with open('../halt', 'w'): 
                pass
            break 

    # identify currently running MDs 
    running_MDs = [md for md in omm_runs if not os.path.exists(md + '/new_pdb')]
    # decide which MD to stop, (no outliers in past 10ns/50ps = 200 frames) 
    n_timeout = time_out / 0.05 
    for md in running_MDs: 
        md_label = os.path.basename(md)
        current_frames = traj_dict[md]
        # low bound for minimal MD runs, 2 * timeout 
        if current_frames > n_timeout * 2: 
            current_outliers = glob(outliers_pdb_path + f"{md_label}_*pdb")
            if current_outliers != []: 
                latest_outlier = current_outliers[-1]
                latest_frame = int(latest_outlier.split('.')[0].split('_')[-1])
                # last 10 ns had outliers 
                if current_frames - latest_frame < n_timeout: 
                    continue 
            restart_pdb = os.path.abspath(restart_pdbs.pop(0))
            with open(md + '/new_pdb', 'w') as fp: 
                fp.write(restart_pdb)

    print(f"=======>Iteration {iteration} done<========")
    iteration += 1


# ## Restarts from check point 
# used_checkpnts = glob(os.path.join(md_path, 'omm_runs_*/omm_runs_*.chk')) 
# restart_checkpnts = [] 
# for checkpnt in checkpnt_list: 
#     checkpnt_filepath = os.path.join(outliers_pdb_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
#     if not os.path.exists(checkpnt_filepath): 
#         shutil.copy2(checkpnt, checkpnt_filepath) 
#         print([os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list]) 
#         # includes only checkpoint of trajectory that contains an outlier 
#         if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list):  
#             restart_checkpnts.append(checkpnt_filepath) 

# Write record for next step 
## 1> restarting checkpoint; 2> unused outliers (ranked); 3> used outliers (shuffled) 
# random.shuffle(used_pdbs) 
# restart_points = restart_checkpnts + restart_pdbs + used_pdbs  
# print(restart_points) 

# restart_points_filepath = os.path.abspath('./restart_points.json') 
# with open(restart_points_filepath, 'w') as restart_file: 
#     json.dump(restart_points, restart_file) 


