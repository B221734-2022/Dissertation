#######################################################################################################
# This script evaluates the scores by both the new and old SCORCH models on an input pre-scored CSV   #
# "scorch_evaluator.py {csv_path} (all - optional)". All scores all poses, blank scores the best only #
#######################################################################################################

import sys
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Load the test data
test_csv_path = sys.argv[1]
test = pd.read_csv(test_csv_path)

# Check if we want all poses or only the highest scoring ones
if len(sys.argv) > 2 and sys.argv[2] == 'all':
    evaluation_type = 'All'
else:
    evaluation_type = 'Highest'

# Create output directory named after the input CSV file and the evaluation type
output_dir = os.path.join('/home/s2451611/MScProject/Model_evaluation/Screening_power/Original_SCORCH_model/', evaluation_type + '_' + os.path.splitext(os.path.basename(test_csv_path))[0])
os.makedirs(output_dir, exist_ok=True)

# Rank the poses based on the "SCORCH_pose_score" scores
test['PoseScoreRank'] = test.groupby('PDBCode')['SCORCH_pose_score'].rank(method='dense', ascending=False)

# Subset the dataframe based on the evaluation type
if evaluation_type == 'All':
    poses_to_evaluate = test
else:
    poses_to_evaluate = test.loc[test.PoseScoreRank == 1]

# Calculate ROC-AUC and AUCPR
roc_auc = roc_auc_score(poses_to_evaluate['Label'], poses_to_evaluate['SCORCH_pose_score'])
aucpr = average_precision_score(poses_to_evaluate['Label'], poses_to_evaluate['SCORCH_pose_score'])

output_csv_path = os.path.join(output_dir, 'predictions.csv')
poses_to_evaluate.to_csv(output_csv_path, index=False)


print(f"ROC-AUC: {roc_auc}\nAUCPR: {aucpr}")

with open(f"{output_dir}/AUCPR_and_ROCAUC.txt", "w") as f:
    f.write(f"ROC-AUC: {roc_auc}\nAUCPR: {aucpr}")

# Plot PR curve
precision, recall, _ = precision_recall_curve(poses_to_evaluate['Label'], poses_to_evaluate['SCORCH_pose_score'])
plt.figure()
plt.plot(recall, precision, label=f'AUCPR = {aucpr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='best')
plt.savefig(os.path.join(output_dir, 'pr_curve.png'))

# Plot ROC curve
fpr, tpr, _ = roc_curve(poses_to_evaluate['Label'], poses_to_evaluate['SCORCH_pose_score'])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve')
plt.legend(loc='best')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
