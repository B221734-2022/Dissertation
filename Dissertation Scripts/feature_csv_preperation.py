###########################################################################################
# This script edits the format of the selected_features_and_labels.csv files to match the #
# SCORCH training data csv files                                                          #
###########################################################################################

import pandas as pd

# Load the CSV file
df = pd.read_csv('/home/s2451611/MScProject/Features/VAL_300_O3A_selected_features_and_labels.csv')

# Rename the 'Receptor' column to 'PDBCode' and keep only the first 4 characters
df['PDBCode'] = df['Receptor'].str[:4]

# Rename the 'Ligand' column to 'Pose name' and replace its content
df['Pose name'] = df['PDBCode'] + '_crystal_pose'

# Drop the old 'Receptor' and 'Ligand' columns
df = df.drop(['Receptor', 'Ligand'], axis=1)

# Reorder the columns to place 'Pose name', 'PDBCode', 'Label' at the end
columns = [col for col in df.columns if col not in ['Pose name', 'PDBCode', 'Label']] + ['Pose name', 'PDBCode', 'Label']
df = df[columns]

# Save the transformed DataFrame to a new CSV file
df.to_csv('/home/s2451611/MScProject/VAL_PREPARED_300_O3A_selected_features_and_labels.csv', index=False)
