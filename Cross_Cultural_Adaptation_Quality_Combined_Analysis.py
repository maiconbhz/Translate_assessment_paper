import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Cores
cor4 = "#FF0000"
cor3 = "#FFA500"
cor2 = "#FFFF00"
cor1 = "#00FF00"

plt.rcParams.update({'font.size': 8})  # Adjust font size if necessary

# Define manual quality assessment data
data_quality_manual = {
    'Study 1': {'Translation': 3, 'Synthesis': 3, 'Back translation': 3, 'Expert committee': 3, 'Prefinal Version': 3},
    'Study 2': {'Translation': 3, 'Synthesis': 0, 'Back translation': 3, 'Expert committee': 2, 'Prefinal Version': 1},
    'Study 3': {'Translation': 3, 'Synthesis': 0, 'Back translation': 3, 'Expert committee': 3, 'Prefinal Version': 1},
    'Study 4': {'Translation': 3, 'Synthesis': 3, 'Back translation': 3, 'Expert committee': 0, 'Prefinal Version': 1},
    'Study 5': {'Translation': 3, 'Synthesis': 1, 'Back translation': 1, 'Expert committee': 1, 'Prefinal Version': 1}
}

data_transparency_manual = {
    'Study 1': {'Translation': 2, 'Synthesis': 2, 'Back translation': 0, 'Expert committee': 2, 'Prefinal Version': 2},
    'Study 2': {'Translation': 0, 'Synthesis': 2, 'Back translation': 2, 'Expert committee': 0, 'Prefinal Version': 1},
    'Study 3': {'Translation': 2, 'Synthesis': 0, 'Back translation': 2, 'Expert committee': 2, 'Prefinal Version': 1},
    'Study 4': {'Translation': 0, 'Synthesis': 2, 'Back translation': 0, 'Expert committee': 2, 'Prefinal Version': 0},
    'Study 5': {'Translation': 2, 'Synthesis': 2, 'Back translation': 2, 'Expert committee': 2, 'Prefinal Version': 0}
}

# Convert to pandas DataFrames
df_quality = pd.DataFrame(data_quality_manual).T
df_transparency = pd.DataFrame(data_transparency_manual).T

# Process data for bar charts
summary_quality = np.array([
    np.sum(df_quality == 3, axis=0),
    np.sum(df_quality == 2, axis=0),
    np.sum(df_quality == 1, axis=0), 
    np.sum(df_quality == 0, axis=0)
])

summary_transparency = np.array([
    np.sum(df_transparency == 2, axis=0),
    np.sum(df_transparency == 1, axis=0),
    np.sum(df_transparency == 0, axis=0),
])

# Define colors
color_map = {3: cor1, 2: cor2, 1: cor3, 0: cor4}
summary_colors = [cor1, cor2, cor3, cor4]
transparency_color_map = {2: cor1, 1: cor2, 0: cor4}
transparency_colors = [cor1, cor2, cor4]

# Define colors FontAwesome icons
quality_icon_map = {
    0: "",
    1: "\uf005",
    2: "\uf005"*2,
    3: "\uf005"*3
}

transparency_icon_map = {
    0: "",
    1: "\uf005",
    2: "\uf005"*2
}

categories = df_quality.columns
studies = df_quality.index
data = df_quality.to_numpy()
transparency_data = df_transparency.to_numpy()

# Define figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
subplot_labels = ["A)", "B)", "C)", "D)"]

# FIGURE 1: Individual Study Quality Matrix
ax1 = axes[0, 0]
for i in range(len(studies)):
    for j in range(len(categories)):
        value = data[i, j]
        ax1.add_patch(plt.Rectangle((j, i), 1, 1, color=color_map[value], ec="black"))
        ax1.text(j + 0.5, i + 0.5, quality_icon_map[value], fontsize=10, ha="center", va="center", color="white", family="Font Awesome 5 Free Solid")

ax1.set_xticks(np.arange(len(categories)) + 0.5)
ax1.set_xticklabels(categories, rotation=45, ha="right")
ax1.set_yticks(np.arange(len(studies)) + 0.5)
ax1.set_yticklabels(studies)
ax1.set_xlim(0, len(categories))
ax1.set_ylim(0, len(studies))
ax1.invert_yaxis()
ax1.set_aspect('equal')
ax1.set_title(f"{subplot_labels[0]} Cross-Cultural Adaptation Quality - Individual Studies", loc='left')

# FIGURE 2: Quality Assessment Summary
ax2 = axes[0, 1]
bottom = np.zeros(len(categories))
for i in range(4):
    ax2.barh(categories, summary_quality[i], left=bottom, color=summary_colors[i], label=["High quality", "Good quality", "Moderate quality", "Low quality"][i])
    bottom += summary_quality[i]

ax2.set_xlabel("Number of studies")
ax2.invert_yaxis()  # Invert Y-axis for Figure 2
ax2.set_title(f"{subplot_labels[1]} Cross-Cultural Adaptation Quality - Consolidated", loc='left')

# FIGURE 3: Transparency Assessment Matrix
ax3 = axes[1, 0]
for i in range(len(studies)):
    for j in range(len(categories)):
        value = transparency_data[i, j]
        ax3.add_patch(plt.Rectangle((j, i), 1, 1, color=transparency_color_map[value], ec="black"))
        ax3.text(j + 0.5, i + 0.5, transparency_icon_map[value], fontsize=10, ha="center", va="center", color="white", family="Font Awesome 5 Free Solid")

ax3.set_xticks(np.arange(len(categories)) + 0.5)
ax3.set_xticklabels(categories, rotation=45, ha="right")
ax3.set_yticks(np.arange(len(studies)) + 0.5)
ax3.set_yticklabels(studies)
ax3.set_xlim(0, len(categories))
ax3.set_ylim(0, len(studies))
ax3.invert_yaxis()
ax3.set_aspect('equal')
ax3.set_title(f"{subplot_labels[2]} Transparency in Cross-Cultural Adaptation - Individual Studies", loc='left')

# FIGURE 4: Transparency Assessment Summary
ax4 = axes[1, 1]
bottom = np.zeros(len(categories))
for i in range(3):
    ax4.barh(categories, summary_transparency[i], left=bottom, color=transparency_colors[i], label=["Yes", "Partially","No"][i])
    bottom += summary_transparency[i]

ax4.set_xlabel("Number of studies")
ax4.invert_yaxis()  # Invert Y-axis for Figure 4
ax4.set_title(f"{subplot_labels[3]} Transparency in Cross-Cultural Adaptation - Consolidated", loc='left')

# Quality Legend
legend_elements_quality = [
    Patch(facecolor=cor1, edgecolor="black", label="★★★ High quality (3)"),
    Patch(facecolor=cor2, edgecolor="black", label="★★ Good quality (2)"),
    Patch(facecolor=cor3, edgecolor="black", label="★ Moderate quality (1)"),
    Patch(facecolor=cor4, edgecolor="black", label="Low quality (0)")
]

# Transparency Legend
legend_elements_transparency = [
    Patch(facecolor=cor1, edgecolor="black", label="★★ Yes - Transparent"),
    Patch(facecolor=cor2, edgecolor="black", label="★ Partially - Transparent"),
    Patch(facecolor=cor4, edgecolor="black", label="No - Not Transparent")
]

legend_quality = fig.legend(
    handles=legend_elements_quality, 
    loc="lower center", 
    ncol=4, 
    fontsize=12, 
    bbox_to_anchor=(0.5, 0.56)  # Ajuste para posicionamento após figuras 1 e 2
)

legend_transparency = fig.legend(
    handles=legend_elements_transparency, 
    loc="lower center", 
    ncol=3, 
    fontsize=12, 
    bbox_to_anchor=(0.5, 0.12)  # Ajuste para posicionamento após figuras 3 e 4
)

plt.tight_layout(rect=[0, 0.16, 1, 1])  # Ajusta espaço para as legendas
fig.subplots_adjust(hspace=0.7)

plt.show()



