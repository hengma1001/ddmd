import os, sys, h5py
import numpy as np 
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .vae_conv import conv_variational_autoencoder


def CVAE(input_shape, latent_dim=3): 
    image_size = input_shape[:-1]
    channels = input_shape[-1]
    conv_layers = 4
    feature_maps = [64,64,32,16]
    filter_shapes = [(3,3),(3,3),(3,3),(3,3)]
    strides = [(1,1),(2,2),(2,2),(1,1)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]

    feature_maps = feature_maps[0:conv_layers];
    filter_shapes = filter_shapes[0:conv_layers];
    strides = strides[0:conv_layers];
    autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
               filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim); 
    # autoencoder.model.summary()
    return autoencoder


def data_split(x, test=False): 
    train, val = train_test_split(x, train_size=.7)
    if test: 
        val, test_data = train_test_split(val, train_size=.5)
        return train, val, test_data
    else: 
        return train, val

    
def run_cvae(gpu_id, cm_file, hyper_dim=3, epochs=100, batch_size=1000, skip=1): 
    # read contact map from h5 file 
    cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
    cm_data_input = cm_h5[u'contact_maps'].value
    cm_data_input = cm_data_input[::skip]

    # splitting data into train and validation
    np.random.shuffle(cm_data_input) 
    train_val_split = int(0.8 * len(cm_data_input))
    cm_data_train, cm_data_val = cm_data_input[:train_val_split], cm_data_input[train_val_split:] 
    input_shape = cm_data_train.shape
    cm_h5.close()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id) 
    
    cvae = CVAE(input_shape[1:], hyper_dim) 
    
#     callback = EmbeddingCallback(cm_data_train, cvae)
    cvae.train(cm_data_train, validation_data=cm_data_val, batch_size = batch_size, epochs=epochs) 
    
    return cvae 
