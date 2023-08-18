########################################################################################################
# This script takes a dataset name input, and produced a PR Curve, ROC Curve, and a bar chart of AUCPR #
# and ROC-AUC for all models.                                                                          #
# "python evaluation_plot_generator.py {dataset name: new_test, dekois, scorch_test}"                  #
########################################################################################################

import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the models and their corresponding CSV file paths for each dataset
models = {
    "SCORCH": {
        "new_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/Original_SCORCH_model/Highest_NEW_TEST_set_Formatted_SCORCH_original_model_output/predictions.csv",
        "scorch_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/Original_SCORCH_model/Highest_SCORCH_TEST_set_Formatted_SCORCH_original_model_output/predictions.csv",
        "dekois": "/home/s2451611/MScProject/Model_evaluation/Screening_power/Original_SCORCH_model/Highest_FIXED_Formatted_DEKOIS_SCORCH_original_model_output/predictions.csv",
        "prediction_column": "SCORCH_pose_score"
    },
    "SCORCH GBDT Model": {
        "new_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/SCORCH_XGBoost_model/Highest_TEST_final_features_and_labels/predictions.csv",
        "scorch_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/SCORCH_XGBoost_model/Highest_495_test_models_58/predictions.csv",
        "dekois": "/home/s2451611/MScProject/Model_evaluation/Screening_power/SCORCH_XGBoost_model/Highest_FIXED_DEKOIS_scaled_selected_features/predictions.csv",
        "prediction_column": "y_pred"
    },
    "CrystalBoost": {
        "new_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_boosters/Highest_TEST_final_features_and_labels/predictions.csv",
        "scorch_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_boosters/Highest_495_test_models_58/predictions.csv",
        "dekois": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_boosters/Highest_FIXED_DEKOIS_scaled_selected_features/predictions.csv",
        "prediction_column": "y_pred"
    },
    "SCORCH with CrystalBoost": {
        "new_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_in_SCORCH/Highest_NEW_TEST_set_Formatted_XGBoost_model_9_in_SCORCH/predictions.csv",
        "scorch_test": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_in_SCORCH/Highest_SCORCH_TEST_set_Formatted_XGBoost_model_9_in_SCORCH/predictions.csv",
        "dekois": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_in_SCORCH/Highest_FIXED_DEKOIS_Formatted_XGBoost_model_9_in_SCORCH/predictions.csv",
        "prediction_column": "SCORCH_pose_score"
    }
}

# Get the dataset name and output directory from the command arguments
dataset_name = sys.argv[1]
output_dir = "/home/s2451611/MScProject/Model_evaluation/Screening_power_plots"

os.makedirs(output_dir, exist_ok=True)

# Function to preprocess data based on dataset and model
def preprocess_data(model_name, data):
    # Determine the appropriate column to rank based on model name
    if model_name in ["SCORCH", "SCORCH with CrystalBoost"]:
        prediction_column = 'SCORCH_pose_score'
    else:
        prediction_column = 'y_pred'
    
    # Rank based on the prediction_column
    data['PoseScoreRank'] = data.groupby('PDBCode')[prediction_column].rank(method='dense', ascending=False)
    
    # Return only the best poses
    return data.loc[data.PoseScoreRank == 1]

# Store AUCPR and ROC-AUC for bar charts
aucpr_values = []
roc_auc_values = []
pr_curves = []
roc_curves = []

# Iterate through models and calculate metrics
pr_curves = []
for model_name, model_info in models.items():
    csv_path = model_info[dataset_name]
    data = pd.read_csv(csv_path)
    data = preprocess_data(model_name, data)
    prediction_column = model_info["prediction_column"]
    labels = data['Label']

    # Calculate ROC-AUC and AUCPR
    roc_auc = roc_auc_score(data['Label'], data[prediction_column])
    aucpr = average_precision_score(data['Label'], data[prediction_column])
    aucpr_values.append(aucpr)
    roc_auc_values.append(roc_auc)

    # ROC curve
    fpr, tpr, _ = roc_curve(data['Label'], data[prediction_column])
    roc_curves.append((fpr, tpr))

    # PR curve
    precision, recall, _ = precision_recall_curve(data['Label'], data[prediction_column])
    pr_curves.append((precision, recall))

# Bar chart parameters
bar_width = 0.5
bar_positions = np.arange(len(models))
colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99']

# ROC-AUC bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(bar_positions, roc_auc_values, width=bar_width, color=colors)
plt.ylabel('ROC-AUC', fontsize=12, fontweight='heavy')
plt.xlabel('Scoring Function', fontsize=12, fontweight='heavy')
plt.title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, max(roc_auc_values) + 0.1)
for i in range(len(bars)):
    plt.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height() + 0.01, f'{roc_auc_values[i]:.2f}', ha='center', va='bottom')
plt.legend(bars, models.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_roc_auc_bar.pdf'))

# AUCPR bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(bar_positions, aucpr_values, width=bar_width, color=colors)
plt.ylabel('AUCPR', fontsize=12, fontweight='heavy')
plt.xlabel('Scoring Function', fontsize=12, fontweight='heavy')
plt.title('AUCPR Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, max(aucpr_values) + 0.1)
for i in range(len(bars)):
    plt.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height() + 0.01, f'{aucpr_values[i]:.2f}', ha='center', va='bottom')
plt.legend(bars, models.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_aucpr_bar.pdf'))

# ROC curve plot
plt.figure(figsize=(14, 6))
for model_name, roc_auc, fpr_tpr in zip(models.keys(), roc_auc_values, roc_curves):
    fpr, tpr = fpr_tpr
    plt.plot(fpr, tpr, label=f'{model_name} ROC AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Random guesser")
plt.xlabel('False Positive Rate', fontsize=14, fontweight='heavy')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='heavy')
plt.title('Receiver Operating Characteristic Curve', fontsize=16, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'weight': 'bold', 'size': 13})  
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_roc_curve.pdf'))



# PR curve plot
plt.figure(figsize=(14, 6))
random_guess = sum(labels) / len(labels)
for model_name, aucpr, prec_recall in zip(models.keys(), aucpr_values, pr_curves):
    precision, recall = prec_recall
    plt.plot(recall, precision, label=f'{model_name} AUCPR = {aucpr:.2f}', linewidth=2)
plt.axhline(y=random_guess, linestyle='--', color='black', label="Random guesser")
plt.xlabel('Recall', fontsize=14, fontweight='heavy')
plt.ylabel('Precision', fontsize=14, fontweight='heavy')
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'weight': 'bold', 'size': 13})  
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_pr_curve.pdf'))

