####################################################################
# This script is used to evaluate docking power of the input model #
# using the CSAR 2014 docking power benchmark                      #    
####################################################################

import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os 

os.makedirs(f"/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_boosters/", exist_ok=True)

# Loading the ligand and near-native pose mapping from the CSV file
ligand_pose_df = pd.read_csv('/home/s2451611/MScProject/Model_evaluation/Docking_power/ligand_near_native_poses.csv')
ligand_pose_dict = dict(zip(ligand_pose_df['PDBCode'], ligand_pose_df['Near_Native_Pose']))

# Function to run predictions using the XGBoost model
def run_xgbscore(df, xgb_path):
    xgbscore = pickle.load(open(xgb_path,'rb'))
    prediction = xgbscore.predict_proba(df)[:, 1]
    return prediction

# Loading the CSV file
file_path = '/home/s2451611/MScProject/Model_evaluation/Docking_power/Formatted_CSAR_2014_features.csv'
data_df = pd.read_csv(file_path)

# Dropping the columns that are not required for prediction
test_mod = data_df.drop(['Pose name','PDBCode'], axis=1)

# Path to the XGBoost model
xgb_path = '/home/s2451611/MScProject/Model_Training/XGBoost_model_9_boosters/final_model.pkl'

# Running predictions
data_df['y_pred'] = run_xgbscore(test_mod, xgb_path)

# Ranking the poses
data_df['PoseScoreRank'] = data_df.groupby('PDBCode')['y_pred'].rank(method='dense', ascending=False)

# Identifying the near-native poses
near_native_poses_rank_list = []
for ligand, pose_num in ligand_pose_dict.items():
    pose_name = f"{ligand}_ligand_pose_{pose_num}" # Corrected pattern
    ligand_df = data_df[data_df['PDBCode'] == ligand]
    matching_pose = ligand_df[ligand_df['Pose name'] == pose_name]
    if matching_pose.empty:
        print(f"Error: No matching pose for ligand {ligand}, pose_name {pose_name}")
        print(ligand_df['Pose name'].head()) # Print the first few pose names for this ligand
        continue
    near_native_poses_rank_list.append(matching_pose['PoseScoreRank'].values[0])

# Calculating the mean rank of the near-native poses
mean_rank_near_native = np.mean(near_native_poses_rank_list)
print(f"Mean Rank of Near Native Pose: {mean_rank_near_native}")

# Save the near_native_poses_rank_list to a file
np.save('/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_boosters/near_native_ranks.npy', near_native_poses_rank_list)

# Saving mean near native rank
with open("/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_boosters/CSAR_2014_benchmark.txt", "w") as f:
    f.write(f"Mean Rank of Near Native Pose: {mean_rank_near_native}")

# Creating a box plot
plt.boxplot(near_native_poses_rank_list, showmeans=True, meanprops={'marker':'D', 'markerfacecolor':'white'})
plt.ylabel('Rank of Near Native Pose')
plt.title('Distribution of Ranks for Near Native Poses')
plt.savefig("/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_boosters/CSAR_2014_benchmark_boxplot.png")

# Saving the predictions and ranks to a new CSV file
data_df.to_csv('CSAR_2014_predictions.csv', index=False)
