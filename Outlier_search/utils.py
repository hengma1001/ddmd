import os 
import numpy as np
import h5py 
import errno 
import MDAnalysis as mda 
from cvae.CVAE import CVAE
from keras import backend as K 
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import LocalOutlierFactor 


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


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


def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
    

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for omm_run in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[omm_run]) < 0: 
            return omm_run, local_frame
        else: 
            local_frame -= int(traj_dict[omm_run])
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb


def get_cvae(model_weight, input_size, hyper_dim=3, gpu_id=0): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    cvae = CVAE(input_size, hyper_dim) 
    cvae.model.load_weights(model_weight)
    return cvae


def predict_from_cvae(model_weight, cm_files, hyper_dim=3, padding=2, gpu_id=0): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    # decoy run to identify cvae input shape 
    cm_h5 = h5py.File(cm_files[0], 'r', libver='latest', swmr=True)
    cm_data = cm_h5[u'contact_maps']
    cvae_input = cm_to_cvae(np.array(cm_data), padding=padding)
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cm_h5.close()
    # load weight 
    cvae.model.load_weights(model_weight)
    traj_dict = {}
    cm_predict = [] 
    for i, cm_file in enumerate(cm_files[:]): 
        # Convert everything to cvae input
        cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
        try:
            cm_data = cm_h5['contact_maps']
        except: 
            continue
        cvae_input = cm_to_cvae(np.array(cm_data), padding=padding)
        cm_h5.close()

        # A record of every trajectory length
        omm_run = os.path.dirname(cm_file)
        traj_dict[omm_run] = cvae_input.shape[0]
        # Get the predicted embeddings 
        embeddings = cvae.return_embeddings(cvae_input) 
        cm_predict.append(embeddings) 
        # if i % 10 == 0: 
        #     print embeddings.shape, i, stamp_to_time(time.time())

    cm_predict = np.vstack(cm_predict) 
    # clean up the keras session
    del cvae 
    K.clear_session()
    return cm_predict, traj_dict


def outliers_from_latent_loc(cm_predict, n_outliers=500, n_jobs=1): 
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=n_jobs).fit(cm_predict) 
    # label = clf.predict(cm_predict) 
    sort_indices = np.argsort(clf.negative_outlier_factor_)[:n_outliers]
    sort_scores = clf.negative_outlier_factor_[sort_indices] 
    return sort_indices, sort_scores


def outliers_from_latent(cm_predict, eps=0.35): 
    db = DBSCAN(eps=eps, min_samples=10).fit(cm_predict)
    db_label = db.labels_
    outlier_list = np.where(db_label == -1)
    return outlier_list


def outliers_largeset(cm_predict, n_outliers=500, n_jobs=1): 
    indices_10 = []
    scores_10 = []
    for i in range(10): 
        cm_predict_loc = cm_predict[i::10] 
        indices_1, score_1 = outliers_from_latent_loc(
                cm_predict_loc, n_outliers=n_outliers, n_jobs=n_jobs) 
        indices_10.append(indices_1 * 10 + 1) 
        scores_10.append(score_1) 
    indices = np.hstack(indices_10) 
    scores = np.hstack(scores_10) 
    ranked_indices = np.argsort(scores)[:n_outliers] 
    ranked_scores = scores[ranked_indices] 
    return ranked_indices, ranked_scores
