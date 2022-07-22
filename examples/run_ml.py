import os 
from ddmd.ml import ml_run

pdb_file = '/lambda_stor/homes/heng.ma/Research/ddmd/ddmd/data/pdbs/bba/1FME-unfolded.pdb'
md_path = './'
runs = ml_run(pdb_file, md_path)
print(runs.get_numberofFrames())
print(runs.get_contact_maps().shape)
# cvae, cvae_setup = runs.train_cvae(epochs=10, strides=[(1, 1), (1, 1), (1, 1), (2, 2)], atom_sel='name N')
runs.ddmd_run(epochs=10, strides=[(1, 1), (1, 1), (1, 1), (2, 2)], cutoff=12, atom_sel='name N')
