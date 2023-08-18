######################################################################################
# This script adds the ligand label to the features csv to include in model training #
######################################################################################

import pandas as pd

# Load the first CSV file
df1 = pd.read_csv('/home/s2451611/MScProject/binding_data/converted_binding_data.csv')

# Load the second CSV file
df2 = pd.read_csv('/home/s2451611/MScProject/VAL_300_O3A_crystal_pose_features.csv')

# Extract the PDBCode from the "Ligand" column in df2
df2['PDBCode'] = df2['Ligand'].str.slice(0, 4)

# Merge the two dataframes based on 'PDBCode' in df1 and 'PDBCode' in df2
merged_df = pd.merge(df2, df1[['PDBCode', 'Label']], on='PDBCode', how='left')
merged_df.drop('PDBCode', axis=1, inplace=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('/home/s2451611/MScProject/Features/VAL_300_O3A_selected_features_and_labels.csv', index=False)
