###################################################################
# This script runs MOAD_pocket_and_ligand_isolator.py in parallel #
###################################################################

from joblib import Parallel, delayed
import subprocess
import os

# The directories where your structure files are located
structure_dirs = ["/home/s2451611/MScProject/Raw_Data/MISSING_BindingMOAD_pdbs/"]


# The directories where you want to save successful and problematic structures
success_dir = "/home/s2451611/MScProject/Raw_Data/MISSING_Binding_MOAD_Successes/"
problem_dir = "/home/s2451611/MScProject/Raw_Data/MISSING_Binding_MOAD_Problems/"

# The location of your binding data csv
binding_data = "/home/s2451611/MScProject/Raw_Data/nr_bind.csv"

script = "/home/s2451611/MScProject/MOAD_pocket_and_ligand_isolator.py"
# Cutoff and exclusion parameters
cutoff = 14
exclusion = 5

# Construct the command for each directory
def run_command(structure_dir):
    cmd = f"python {script} -loc {structure_dir} -suc {success_dir} -prob {problem_dir} -ref {binding_data} -cutoff {cutoff} -exclusion {exclusion}"
    subprocess.run(cmd, shell=True)

# Use joblib to run the commands in parallel
Parallel(n_jobs=-1)(delayed(run_command)(structure_dir) for structure_dir in structure_dirs)
