###############################################################################################################
# This script runs my modified scorch.py on all receptor-ligand pairs. The scorch.py is modified to only save # 
# a csv containing all selected features from SCORCH                                                          #
###############################################################################################################

import os
import subprocess
import pandas as pd
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

os.makedirs("/home/s2451611/MScProject/temp/", exist_ok=True)

def process_pdb_id(pdb_id):
    base_dir = "/home/s2451611/MScProject/300_O3A_MMFF_Split_data/VAL"
    scorch_dir = "/home/s2451611/MScProject/SCORCH"

    ligand_path = os.path.join(base_dir, pdb_id, f"{pdb_id}_aligned.pdbqt") # REMEMBER TO EDIT THIS BASE ON PDBQT FILE NAME FORMAT
    receptor_path = os.path.join(base_dir, pdb_id, f"{pdb_id}_receptor.pdbqt")

    os.chdir(scorch_dir)

    cmd = f"python scorch.py -r {receptor_path} -l {ligand_path} -o /home/s2451611/MScProject/temp/{pdb_id}.csv"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        with open("openbabel_error_com.txt", "a") as file:
            file.write(f"Error with pdb_id: {pdb_id}")
            file.write(f"Receptor path: {receptor_path}")
            file.write(f"Ligand path: {ligand_path}")
            file.write(f"Error message: {stderr.decode()}")


# get the list of pdb_ids
pdb_ids = os.listdir("/home/s2451611/MScProject/300_O3A_MMFF_Split_data/VAL")

# run in parallel with progress bar
Parallel(n_jobs=-1)(delayed(process_pdb_id)(pdb_id) for pdb_id in tqdm(pdb_ids))

# merge all temporary csv files
temp_files = glob.glob('/home/s2451611/MScProject/temp/*.csv')

df_list = []
for filename in temp_files:
    df = pd.read_csv(filename)
    if isinstance(df, pd.DataFrame):  # check if df is a DataFrame
        df_list.append(df)
    else:
        print(f"Error reading file: {filename}")

# Concatenate all dataframes together
full_df = pd.concat(df_list)

# Write the final dataframe to file
full_df.to_csv('/home/s2451611/MScProject/VAL_300_O3A_crystal_pose_features.csv', index=False)

# Optionally, remove the temporary files
for filename in temp_files:
    os.remove(filename)




# SINGLE THREADED VERSION #

# import os
# import subprocess
# from tqdm import tqdm

# def process_pdb_id(pdb_id):
#     base_dir = "/home/s2451611/MScProject/Split_data/TRAIN"
#     scorch_dir = "/home/s2451611/SCORCH"

#     ligand_path = os.path.join(base_dir, pdb_id, f"{pdb_id}_aligned.pdbqt")
#     receptor_path = os.path.join(base_dir, pdb_id, f"{pdb_id}_receptor.pdbqt")

#     os.chdir(scorch_dir)

#     cmd = f"python scorch.py -r {receptor_path} -l {ligand_path} -o ~/MScProject/scorch_out.csv"
#     subprocess.run(cmd, shell=True)

# # get the list of pdb_ids
# pdb_ids = os.listdir("/home/s2451611/MScProject/Split_data/TRAIN")

# # iterate over each pdb_id, calling the function on it
# for pdb_id in tqdm(pdb_ids):
#     process_pdb_id(pdb_id)

