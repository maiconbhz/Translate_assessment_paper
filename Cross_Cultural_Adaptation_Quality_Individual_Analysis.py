import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define colors for different quality levels
color_high_quality = "#00FF00"  # Green (High quality)
color_good_quality = "#FFFF00"  # Yellow (Good quality)
color_moderate_quality = "#FFA500"  # Orange (Moderate quality)
color_low_quality = "#FF0000"  # Red (Low quality)

# Define manual quality assessment data
quality_data = {
    '': {'Translation': 3, 'Synthesis': 3, 'Back translation': 3, 'Expert committee': 1, 'Prefinal Version': 0}
}

# Define manual transparency assessment data
transparency_data = {
    '': {'Translation': 2, 'Synthesis': 2, 'Back translation': 2, 'Expert committee': 2, 'Prefinal Version': 0}
}

# Convert to pandas DataFrames
quality_df = pd.DataFrame(quality_data).T
transparency_df = pd.DataFrame(transparency_data).T

# Define color and icon mappings for quality assessment
quality_color_map = {3: color_high_quality, 2: color_good_quality, 1: color_moderate_quality, 0: color_low_quality}
quality_icon_map = {0: "", 1: "\uf005", 2: "\uf005"*2, 3: "\uf005"*3}  # Font Awesome stars

# Define color and icon mappings for transparency assessment
transparency_color_map = {2: color_high_quality, 1: color_good_quality, 0: color_low_quality}
transparency_icon_map = {0: "", 1: "\uf005", 2: "\uf005"*2}  # Font Awesome stars

# Extract column names (categories) and row indices (studies)
categories = quality_df.columns
studies = quality_df.index
quality_matrix = quality_df.to_numpy()
transparency_matrix = transparency_df.to_numpy()

# FIGURE 1: Individual Quality Assessment Matrix
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i in range(len(studies)):
    for j in range(len(categories)):
        value = quality_matrix[i, j]
        ax1.add_patch(plt.Rectangle((j, i), 1, 1, color=quality_color_map[value], ec="black"))
        ax1.text(j + 0.5, i + 0.5, quality_icon_map[value], fontsize=12, ha="center", va="center", color="white", family="Font Awesome 5 Free Solid")

# Configure x-axis labels
ax1.set_xticks(np.arange(len(categories)) + 0.5)
ax1.set_xticklabels(categories, rotation=45, ha="right")

# Configure y-axis labels
ax1.set_yticks(np.arange(len(studies)) + 0.5)
ax1.set_yticklabels(studies)
ax1.set_xlim(0, len(categories))
ax1.set_ylim(0, len(studies))
ax1.invert_yaxis()
ax1.set_aspect('equal')
ax1.set_title("Cross-Cultural Adaptation Quality")

# Define legend for quality assessment
legend_elements_quality = [
    Patch(facecolor=color_high_quality, edgecolor="black", label="★★★ High quality (3)"),
    Patch(facecolor=color_good_quality, edgecolor="black", label="★★ Good quality (2)"),
    Patch(facecolor=color_moderate_quality, edgecolor="black", label="★ Moderate quality (1)"),
    Patch(facecolor=color_low_quality, edgecolor="black", label="Low quality (0)")
]
fig1.tight_layout(rect=[0, 0.1, 1, 1])
fig1.legend(handles=legend_elements_quality, loc="lower center", bbox_to_anchor=(0.5, 0.16), ncol=4)

plt.show()

# FIGURE 2: Transparency Assessment Matrix
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i in range(len(studies)):
    for j in range(len(categories)):
        value = transparency_matrix[i, j]
        ax2.add_patch(plt.Rectangle((j, i), 1, 1, color=transparency_color_map[value], ec="black"))
        ax2.text(j + 0.5, i + 0.5, transparency_icon_map[value], fontsize=12, ha="center", va="center", color="white", family="Font Awesome 5 Free Solid")

# Configure x-axis labels
ax2.set_xticks(np.arange(len(categories)) + 0.5)
ax2.set_xticklabels(categories, rotation=45, ha="right")

# Configure y-axis labels
ax2.set_yticks(np.arange(len(studies)) + 0.5)
ax2.set_yticklabels(studies)
ax2.set_xlim(0, len(categories))
ax2.set_ylim(0, len(studies))
ax2.invert_yaxis()
ax2.set_aspect('equal')
ax2.set_title("Cross-Cultural Adaptation Transparency")

# Define legend for transparency assessment
legend_elements_transparency = [
    Patch(facecolor=color_high_quality, edgecolor="black", label="★★ Yes - Transparent"),
    Patch(facecolor=color_good_quality, edgecolor="black", label="★ Partially - Transparent"),
    Patch(facecolor=color_low_quality, edgecolor="black", label="No - Not Transparent")
]
fig2.tight_layout(rect=[0, 0.1, 1, 1])
fig2.legend(handles=legend_elements_transparency, loc="lower center", bbox_to_anchor=(0.5, 0.16), ncol=3)

plt.show()
