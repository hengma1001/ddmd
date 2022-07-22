import os
import time
from ddmd.sim import omm_sim


pdb_file = os.path.abspath('../data/pdbs/bba/1FME-unfolded.pdb')

sim_imp = omm_sim(pdb_file, sim_time=10, explicit_sol=False)
sim_imp.ddmd_run(iter=3)
