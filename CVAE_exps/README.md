# Run CVAE 
```
python train_cvae.py -f cvae_input.h5
```

## Installation on SUMMIT: 
1. TensorFlow gpu version from Anaconda
    ```
    conda install tensorflow-gpu 
    ```
2. Keras from pip
    ```
    pip install keras==2.2.4
    ```
Note: So the only 1.2.1 version of tensorflow is available on powerPC from Anaconda at the moment (8/13/2020). The newest version of Keras doesn't work with older version of TF. 
