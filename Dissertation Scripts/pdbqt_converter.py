################################################################################################
# THIS IS A PYTHON 2 SCRIPT that is designed to convert pdb to pdbqt files with MGLTOOLS 1.5.7 #
################################################################################################

import os
import subprocess
from joblib import Parallel, delayed

# Directory containing the input PDB files
input_dir = "/home/s2451611/MScProject/Aligned_pdbs/300_MCS_UFF_aligned_pdbs"

# Directory to save the output PDBQT files
output_dir = "/home/s2451611/MScProject/Aligned_pdbqt/300_MCS_UFF_pdbqt_aligned_ligands"

# Path to the MGLTools script
prepare_ligand_script = "/home/s2451611/anaconda3/envs/mgltools/bin/prepare_ligand4.py"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the MGLTools script
def run_script(input_file, output_file, pdb_id):
    # Save the current working directory
    cwd = os.getcwd()
    
    # Change to the input file directory
    os.chdir(os.path.dirname(input_file))

    command = "python2 %s -l %s -A hydrogens -o %s -U nphs" % (prepare_ligand_script, os.path.basename(input_file), os.path.abspath(output_file))
    print("Processing file: %s" % input_file)
    print("%s is being processed." % pdb_id)
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print("Processing failed for PDB ID: %s" % pdb_id)
        
    # Change back to the original working directory
    os.chdir(cwd)



# Build up the arguments for each task
tasks = []
for file_name in os.listdir(input_dir):
    if file_name.endswith("_best.pdb"):
        # Get the PDB ID
        pdb_id = file_name.split('_')[0]

        # Prepare the output file name
        output_file = os.path.join(output_dir, "%s_best.pdbqt" % pdb_id)

        # Input file path
        input_file = os.path.join(input_dir, file_name)

        tasks.append((input_file, output_file, pdb_id))

# Run the tasks in parallel
Parallel(n_jobs=-1)(delayed(run_script)(input_file, output_file, pdb_id) for input_file, output_file, pdb_id in tasks)
