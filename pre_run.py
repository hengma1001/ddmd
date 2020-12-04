import os, glob 
import sys 
import shutil
import time

if len(sys.argv) > 1: 
    status = sys.argv[1] 
else: 
    status = 'fail'

# os.system("ps -f -u $USER | grep rp.pmgr_launching.0 | grep -v grep | cut -c 9-15 | xargs -n 1 -t kill")
print(status )
omm_dirs = glob.glob('MD_exps/omm_runs*') 
cvae_dirs = glob.glob('CVAE_exps/cvae_runs_*') 
jsons = glob.glob('Outlier_search/*json') 

result_save = os.path.join('./results', 'result_%d_%s' % (int(time.time()), status)) 
os.makedirs(result_save) 

omm_save = os.path.join(result_save, 'omm_results') 
os.makedirs(omm_save) 
for omm_dir in omm_dirs: 
    shutil.move(omm_dir, omm_save) 

cvae_save = os.path.join(result_save, 'cvae_results') 
os.makedirs(cvae_save) 
for cvae_dir in cvae_dirs: 
    shutil.move(cvae_dir, cvae_save) 

outlier_save = os.path.join(result_save, 'outlier_save/') 
os.makedirs(outlier_save) 
for json in jsons:  
    shutil.move(json, outlier_save) 

if os.path.isdir('Outlier_search/outlier_pdbs'): 
    shutil.move('Outlier_search/outlier_pdbs', outlier_save) 



