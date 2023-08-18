
#####################################################################################
# This script was taken from Miles' XGBScore gitub Lab Notebook. It is used to      #
# download pdb files based on the biounit files contained in BindingMOAD as biounit #
# files are incomplete structures                                                   #
######################################################################################

import os
import requests
from tqdm import tqdm

def create_target_url(filepath):
    
    # get the pdb code from the filename
    target_name = filepath.split('.')[0]
    
    # set the url string using the pdb code as where to download the pdb file
    target_url = f'https://files.rcsb.org/download/{target_name}.pdb'
    return target_url, target_name

def save_target_file(url):
    
    # get the file url and the target name
    url, target_name = create_target_url(url)
    
    # change this as to where you need to save the pdb files
    target_path = f'/home/s2451611/MScProject/BindingMOAD_pdbs/{target_name}.pdb'
    response = requests.get(url)
    
    # ping the pdb and download the file if the url exists
    if response.status_code == 200:
        with open(target_path, 'wb') as file:
            file.write(response.content)
            file.close()
    else:
        print(f'Whoops! Somethings wrong: Response {response.status_code}')


# get the list of all the Binding_MOAD extracted protein-ligand complex pdb codes with no duplicates
filepaths = list(set(list([file.split('.')[0] for file in os.listdir('/home/s2451611/MScProject/BindingMOAD_2020')])))

# download the pdb files for all the Binding_MOAD extracted protein-ligand complex biounit files
with tqdm(total=len(filepaths)) as pbar:
    for target_file in filepaths:
        save_target_file(target_file)
        pbar.update(1)