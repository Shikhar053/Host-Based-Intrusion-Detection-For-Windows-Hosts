import matplotlib.pyplot as plt
import numpy as np

# Techniques
techniques = [
    "Ours(XGBoost)", 
    "Ours (AdaBoost)", 
    "Ours (LightGBM)", 
    "2025: Stacking ensemble+TF-IDF BoW (Without  Mutual Induction Feature Selection Technique)", 
    "2025: Stacking ensemble+TF-IDF BoW (With Mutual Induction Feature Selection Technique)", 
    "2023: Stacking Ensemble-based HIDS.",
    "2021: TF-IDF and SVD-based HIDS."
]

# Metrics: Accuracy, Precision, Recall, F1-Score (as percentages)
data = {
    "Accuracy":    [90.05, 86.50, 88.54, 91.63, 85.27, 87.3, 86.3],
    "Precision":   [86.61, 82.96, 84.95, 87.54, 79.08, 83.2, 86.54],
    "Recall":      [94.78, 91.89, 93.68, 95.65, 94.33, 92.6, 93.57],
    "F1-Score":    [90.51, 87.19, 89.10, 91.42, 86.03, 88.0, 89.92]
}

# Added your final best-performing models
final_models = {
    "XGBoost": [90.05, 86.61, 94.78, 90.51],  # Untuned XGBoost
    "AdaBoost": [86.50, 82.96, 91.89, 87.19],  # Tuned AdaBoost
    "LightGBM": [88.54, 84.95, 93.68, 89.10],  # Tuned LightGBM
}

# Combine your final models data into the dictionary
data["Accuracy"] = [final_models["XGBoost"][0], final_models["AdaBoost"][0], final_models["LightGBM"][0], 91.63, 85.27, 87.3, 86.3]
data["Precision"] = [final_models["XGBoost"][1], final_models["AdaBoost"][1], final_models["LightGBM"][1], 87.54, 79.08, 83.2, 86.54]
data["Recall"] = [final_models["XGBoost"][2], final_models["AdaBoost"][2], final_models["LightGBM"][2], 95.65, 94.33, 92.6, 93.57]
data["F1-Score"] = [final_models["XGBoost"][3], final_models["AdaBoost"][3], final_models["LightGBM"][3], 91.42, 86.03, 88.0, 89.92]

# Plot setup
metrics = list(data.keys())
x = np.arange(len(metrics))  # the label locations
width = 0.12  # width of the bars

fig, ax = plt.subplots(figsize=(14, 6))

# Plot each technique's values
for i, technique in enumerate(techniques):
    offset = (i - len(techniques)/2) * width + width/2
    scores = [data[m][i] for m in metrics]
    bars = ax.bar(x + offset, scores, width, label=technique)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Final plot adjustments
ax.set_ylabel('Score (%)')
ax.set_ylim(80, 97)
ax.set_yticks(np.arange(80, 98, 1))
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title('Comparison of HIDS Techniques on ADFA-WD Dataset')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)
ax.grid(True, linestyle='--', linewidth=0.5, axis='y', alpha=0.7)

plt.tight_layout()
plt.show()
