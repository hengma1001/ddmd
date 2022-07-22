import os 
from ddmd.inference import inference_run

pdb_file = '/lambda_stor/homes/heng.ma/Research/ddmd/ddmd/data/pdbs/bba/1FME-unfolded.pdb'
md_path = './'
vae_path = './'
runs = inference_run(pdb_file, md_path, vae_path)
runs.ddmd_run()
