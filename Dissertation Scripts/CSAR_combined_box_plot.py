#################################################################################################################
# This script makes a combined box plot for the SCORCH XGBoost model and XGBoost_model_9 CSAR benchmark results #
# as well as the SCORCH original and XGBoost_model_9 in SCORCH results                                          #
#################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("/home/s2451611/MScProject/Model_evaluation/Docking_power/", exist_ok=True)

# Load the near_native_poses_rank_list from the saved files
near_native_poses_rank_list_1 = np.load('/home/s2451611/MScProject/Model_evaluation/Docking_power/Original_SCORCH_model/near_native_ranks.npy')
near_native_poses_rank_list_2 = np.load('/home/s2451611/MScProject/Model_evaluation/Docking_power/SCORCH_XGBoot_model/SCORCH_near_native_ranks.npy')
near_native_poses_rank_list_3 = np.load('/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_boosters/near_native_ranks.npy')
near_native_poses_rank_list_4 = np.load('/home/s2451611/MScProject/Model_evaluation/Docking_power/XGBoost_model_9_in_SCORCH/near_native_ranks.npy')

# Combine the four lists into a single data structure for plotting
data_to_plot = [near_native_poses_rank_list_1, near_native_poses_rank_list_2, near_native_poses_rank_list_3, near_native_poses_rank_list_4]

# Create a figure instance
fig, ax = plt.subplots(figsize=(12, 6))

# Create the boxplot with custom colors, small black dots for fliers, and minimalistic whiskers
bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True, meanprops={'marker':'D', 'markerfacecolor':'red', 'markeredgecolor':'black'},
                medianprops={'color': 'black'}, flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4},
                whiskerprops={'color': 'grey', 'linestyle': 'dashed'})

# Set colors for the boxplots (add more colors if needed)
colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add labels with increased size and bold text
ax.set_xticklabels(['SCORCH', 'SCORCH GBDT Model', 'CrystalBoost', 'SCORCH CrystalBoost'])
plt.ylabel('Rank of Near Native Pose', fontsize=14, fontweight='heavy')
plt.xlabel('Scoring Function', fontsize=14, fontweight='heavy')
plt.title('CSAR Benchmark 2014', fontsize=18, fontweight='heavy')

# Print mean value to the right of the mean diamond symbol, with increased text size
for i, line in enumerate(bp['means']):
    x, y = line.get_xydata()[0]
    plt.text(x + 0.1, y, f'{y:.2f}', horizontalalignment='center', color='black', fontsize=12)

# Add legend to explain symbols
legend_elements = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='Mean'),
                   plt.Line2D([0], [0], color='black', lw=1.5, label='Median')]
ax.legend(handles=legend_elements, loc='upper right')

# Save the figure
plt.savefig('/home/s2451611/MScProject/Model_evaluation/Docking_power/All_models_combined_boxplot.pdf')


