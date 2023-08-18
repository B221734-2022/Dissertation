###############################################################################################
# This script is used to evaluate SCORCH XGBoost GDBT model on an input dataset.              #
# It takes a csv filepath as an argument, performs prediction, saves a new csv file with a    #
# prediction column, and plots and calculates AUCPR and ROC-AUC on only the highest rank pose #
# for each ligand defined by PDB id.                                                          #
###############################################################################################

import sys
import os
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Here's the function to run prediction with your XGBoost model
def run_xgbscore(df, xgb_path):
    prediction = model_new.predict_proba(df)[:, 1]
    return prediction

# Define the path to your XGBoost model
with open('/home/s2451611/MScProject/SCORCH/utils/models/xgboost_models/495_models_58_booster.pkl', 'rb') as file:
        model = pickle.load(file)

    
model_new = xgb.XGBClassifier()

# Set the Booster of the new model to be the loaded Booster
model_new._Booster = model

# Load the test data
test_csv_path = sys.argv[1]
test = pd.read_csv(test_csv_path)

# Prepare the test data
test_mod = test.drop(['Pose name','PDBCode','Label'], axis=1)

# Run prediction
test['y_pred'] = run_xgbscore(test_mod, model_new)

# Check if evaluating all poses or only the highest scoring ones
if len(sys.argv) > 2 and sys.argv[2] == 'all':
    evaluation_type = 'All'
else:
    evaluation_type = 'Highest'

# Create output directory named after the input CSV file and the evaluation type
output_dir = os.path.join('/home/s2451611/MScProject/Model_evaluation/Screening_power/SCORCH_XGBoost_model/', evaluation_type + '_' + os.path.splitext(os.path.basename(test_csv_path))[0])
os.makedirs(output_dir, exist_ok=True)

# Save the test data with the predictions
output_csv_path = os.path.join(output_dir, 'predictions.csv')
test.to_csv(output_csv_path, index=False)

# Rank the poses based on the prediction scores
test['PoseScoreRank'] = test.groupby('PDBCode')['y_pred'].rank(method='dense', ascending=False)

# Subset the dataframe based on the evaluation type
if evaluation_type == 'All':
    poses_to_evaluate = test
else:
    poses_to_evaluate = test.loc[test.PoseScoreRank == 1]

# Calculate ROC-AUC and AUCPR
roc_auc = roc_auc_score(poses_to_evaluate['Label'], poses_to_evaluate['y_pred'])
aucpr = average_precision_score(poses_to_evaluate['Label'], poses_to_evaluate['y_pred'])

print(f"ROC-AUC: {roc_auc}\nAUCPR: {aucpr}")

with open(f"{output_dir}/AUCPR_and_ROCAUC.txt", "w") as f:
    f.write(f"ROC-AUC: {roc_auc}\nAUCPR: {aucpr}")

# Plot PR curve
precision, recall, _ = precision_recall_curve(poses_to_evaluate['Label'], poses_to_evaluate['y_pred'])
plt.figure()
plt.plot(recall, precision, label=f'AUCPR = {aucpr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='best')
plt.savefig(os.path.join(output_dir, 'pr_curve.png'))

# Plot ROC curve
fpr, tpr, _ = roc_curve(poses_to_evaluate['Label'], poses_to_evaluate['y_pred'])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve')
plt.legend(loc='best')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
