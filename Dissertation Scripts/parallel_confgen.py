#############################################################################################
# This script is used to run the "rdk_confgen.py" script in parallel to generate conformers #
#############################################################################################

from rdkit import Chem
import pandas as pd
import os
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm

output_dir = "/home/s2451611/MScProject/Conformers/300_2023_UFF_conformer_dir"

def confgen(row):
    # Get SMILES string and PDBCode from the row
    smiles_string = row['RDKitCanonSmile']
    pdb_code = row['PDBCode']

    output_file_name = f"{pdb_code}_SMILE.sdf"
    output_file = os.path.join(output_dir, output_file_name)
    
    # Prepare the command
    command = ["confgen", "-i", smiles_string, "-o", output_file, "--numconf", "300"]
    print(f"Running command: {' '.join(command)}")
    
    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # Print output and error messages if any
    if result.stdout:
        print("Command output:", result.stdout.decode())
    if result.stderr:
        print("Command error:", result.stderr.decode())

    # Check for execution errors
    if result.returncode != 0:
        print(f"Command failed with return code: {result.returncode}")

def main():
    # Define the input file paths
    active_smiles_path = r"/home/s2451611/MScProject/SCORCH_SMILES/RDKIT_2023_Active_SMILES.csv"

    # Read the CSV files
    active_df = pd.read_csv(active_smiles_path, dtype={'PDBCode': str})

    # Process the active ligands
    os.makedirs(output_dir, exist_ok=True)
    Parallel(n_jobs=-1)(delayed(confgen)(row) for _, row in tqdm(active_df.iterrows(), total=active_df.shape[0]))

if __name__ == "__main__":
    main()
