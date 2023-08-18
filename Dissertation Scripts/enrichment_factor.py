#############################################################################################
# This script caluclates enrichment factor at various thresholds with ODDT                  #
# Command: enrichment_factor.py (scorch, scorch_xgboost, crystalboost, scorch_crystalboost) #                                                                         #
#############################################################################################

import os
import argparse
import pandas as pd
from oddt.metrics import enrichment_factor
import matplotlib.pyplot as plt
import numpy as np

FILE_PATHS = {
    "SCORCH": "/home/s2451611/MScProject/Model_evaluation/Screening_power/Original_SCORCH_model/Highest_FIXED_Formatted_DEKOIS_SCORCH_original_model_output/predictions.csv",
    "CrystalBoost": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_boosters/Highest_FIXED_DEKOIS_scaled_selected_features/predictions.csv",
    "SCORCH GBDT": "/home/s2451611/MScProject/Model_evaluation/Screening_power/SCORCH_XGBoost_model/Highest_FIXED_DEKOIS_scaled_selected_features/predictions.csv",
    "SCORCH with CrystalBoost": "/home/s2451611/MScProject/Model_evaluation/Screening_power/XGBoost_model_9_in_SCORCH/Highest_FIXED_DEKOIS_Formatted_XGBoost_model_9_in_SCORCH/predictions.csv"
}

PREDICTION_COLUMNS = {
    "SCORCH": "SCORCH_pose_score",
    "CrystalBoost": "y_pred",
    "SCORCH GBDT": "y_pred",
    "SCORCH with CrystalBoost": "SCORCH_pose_score"
}

def calculate_ef(predictions_path, model_name):
    df = pd.read_csv(predictions_path)
    prediction_column = PREDICTION_COLUMNS[model_name]

    # Ensure only the best pose is selected for each PDBCode
    df = df.loc[df.groupby("PDBCode")[prediction_column].idxmax()]

    df_sorted = df.sort_values(by=prediction_column, ascending=False)
    y_true = df_sorted['Label'].values
    y_score = df_sorted[prediction_column].values
 
    desired_thresholds = [0.005, 0.01, 0.02, 0.05]
    ef_results = {}
    for threshold in desired_thresholds:
        ef = enrichment_factor(y_true, y_score, percentage=threshold*100)
        print(f"EF for {model_name} at {threshold*100}%: {ef}")
        ef_results[f'EF_{int(threshold*100)}%'] = ef
    return ef_results



def gather_ef_for_all_models():
    results = {}
    for model in FILE_PATHS.keys():
        results[model] = calculate_ef(FILE_PATHS[model], model)
    return results

def generate_combined_boxplot(ef_results):
    thresholds = ['EF_0%', 'EF_1%', 'EF_2%', 'EF_5%']
    thresholds_lables = ['EF 0.5%', 'EF 1%', 'EF 2%', 'EF 5%']
    colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99']
    fig, ax = plt.subplots(figsize=(12, 6))
    position = np.arange(len(thresholds))
    width = 0.2
    
       
    for idx, model in enumerate(FILE_PATHS.keys()):
        data_to_plot = [ef_results[model][threshold] for threshold in thresholds]
        bars = ax.bar(position - width/2 + idx*width, data_to_plot, width, color=colors[idx], label=model)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    
    ax.set_xticks(position + 0.2)  # Shift x-ticks to the right
    ax.set_xticklabels(thresholds_lables)
    plt.ylabel('Enrichment Factor', fontsize=14, fontweight='heavy')
    plt.xlabel('Thresholds', fontsize=14, fontweight='heavy')
    plt.title('Enrichment Factors for Different Models', fontsize=18, fontweight='heavy')
    plt.legend()
    plt.savefig('/home/s2451611/MScProject/Model_evaluation/DEKOIS_Screening_power_plots_combined_ef_boxplot.pdf')

ef_results = gather_ef_for_all_models()
generate_combined_boxplot(ef_results)

