#######################################################################################
# This script uses an MCS approach and rdkit to align the conformers                  #
# to the  crystal ligand samples using sdf file inputs for both ligands being aligned #
#######################################################################################

import os
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdFMCS
from joblib import Parallel, delayed
from tqdm import tqdm

crystal_dir = "/home/s2451611/MScProject/Raw_Data/openbabel_sdf_crystal_pose_ligands_for_UFF_confs"
conformers_dir = "/home/s2451611/MScProject/Conformers/300_2023_UFF_conformer_dir"
output_dir = "/home/s2451611/MScProject/300_MCS_UFF_aligned_pdbs"
rmsd_file = "/home/s2451611/MScProject/300_MCS_UFF_rdkit_full.txt"
fail_log_file = "/home/s2451611/MScProject/failed_alignment.txt"

# Check if the directories exist, if not create them
directories = [crystal_dir, conformers_dir, output_dir]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

def process_pdb_id(pdb_id):
    try:
        # Read the crystal pose and the conformers
        crystal_pose_file = os.path.join(crystal_dir, f"{pdb_id}_ligand.sdf")
        crystal_pose_supplier = Chem.SDMolSupplier(crystal_pose_file)
        if not crystal_pose_supplier:
            return []
        crystal_pose = crystal_pose_supplier[0]
        if crystal_pose is None:
            return []

        conformers_file = os.path.join(conformers_dir, f"{pdb_id}_SMILE.sdf")
        conformers_supplier = Chem.SDMolSupplier(conformers_file)
        if not conformers_supplier:
            return []
        conformers = [m for m in conformers_supplier if m is not None]
        if not conformers:
            return []

        # Find the MCS
        mcs = rdFMCS.FindMCS([crystal_pose] + list(conformers), 
                            threshold=0.8, 
                            completeRingsOnly=True, 
                            ringMatchesRingOnly=True)

        # Align each conformer to the crystal pose
        patt = Chem.MolFromSmarts(mcs.smartsString)
        refMol = crystal_pose
        refMatch = refMol.GetSubstructMatch(patt)

        rmsVs = []
        for probeMol in conformers:
            mv = probeMol.GetSubstructMatch(patt)
            rms = AllChem.AlignMol(probeMol, refMol, atomMap=list(zip(mv, refMatch)))
            rmsVs.append((rms, pdb_id))

        if rmsVs: # Check if the list is not empty
            # Save the conformer with the lowest RMSD to a pdb file
            min_rmsd_index = rmsVs.index(min(rmsVs))  # Get the index of the conformer with lowest RMSD
            best_conformer = conformers[min_rmsd_index]
        
            writer = Chem.PDBWriter(os.path.join(output_dir, f"{pdb_id}_best.pdb"))
            writer.write(best_conformer)
            writer.close()

        return rmsVs

    except Exception as e:
        with open(fail_log_file, 'a') as fail_log:
            fail_log.write(f"An error occurred while processing PDB ID {pdb_id}: {str(e)}\n")
        return []

pdb_ids = [filename.split('_')[0] for filename in os.listdir(crystal_dir)]

results = Parallel(n_jobs=-1)(delayed(process_pdb_id)(pdb_id) for pdb_id in tqdm(pdb_ids))

all_rmsds = [rms for sublist in results for rms in sublist]

# Calculate average RMSD across all conformers and save to file
avg_rmsd = sum(rmsd for rmsd, pdb_id in all_rmsds) / len(all_rmsds)

with open(rmsd_file, 'w') as f:
    f.write(f"Average RMSD: {avg_rmsd}\n\n")
    for rmsd, pdb_id in all_rmsds:
        f.write(f"RMSD for PDB ID {pdb_id}: {rmsd}\n")
