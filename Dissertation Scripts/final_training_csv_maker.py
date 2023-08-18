#########################################################################################
# This script checks that the prepared, labelled and feature selected csv has identical #
# column names to the SCORCH data csv, then merges the two if it is                     #
#########################################################################################

import pandas as pd

# Load the CSV files
df1 = pd.read_csv('/home/s2451611/MScProject/VAL_PREPARED_300_O3A_selected_features_and_labels.csv')
df2 = pd.read_csv('/home/s2451611/MScProject/SCORCH_Model_Training_Data/495_val_models_58.csv')

# Check if the columns in df1 are the same as in df2
columns_same = set(df1.columns) == set(df2.columns)

print("Are the column names identical?: ", columns_same)

# If the columns are the same, merge the dataframes
if columns_same:
    df_combined = pd.concat([df1, df2])

    # Save the combined DataFrame to a new CSV file
    df_combined.to_csv('TRAIN_O3A_300_final_features_and_labels.csv', index=False)
else:
    print("The dataframes have different columns and can't be merged.")
