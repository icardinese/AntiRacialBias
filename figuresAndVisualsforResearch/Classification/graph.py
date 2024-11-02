import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# First, load or create the DataFrame
recidivism_overall_metrics = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Odds',
        'Your Model w/ Debiasing + TPR Parity',
        'COMPAS Trained Model w/o Debiasing',
        'COMPAS Raw Model w/o Debiasing'
    ],
    'Accuracy (%)': [
        82.639885,
        82.840746,
        81.692970,
        82.324247,
        63.654561,
        64.369460
    ],
    'Precision (%)': [
        82.926242,
        82.980019,
        None,
        None,
        63.787800,
        None
    ],
    'Recall (%)': [
        82.639885,
        82.840746,
        None,
        None,
        63.654561,
        None
    ],
    'F1 Score (%)': [
        82.571939,
        82.801245,
        None,
        None,
        63.639699,
        None
    ]
})

# Melt the DataFrame for easier plotting
metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
df_melted = recidivism_overall_metrics.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

# Remove rows with None values
df_melted = df_melted.dropna(subset=['Value'])

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
plt.title('Recidivism Classification Overall Metrics')
plt.ylabel('Percentage')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Load or create the DataFrame
# (Assuming recidivism_group_metrics is already created as per your data)
recidivism_group_metrics = pd.DataFrame({
    'Model': [
        # Your Model w/o Debiasing
        'Your Model w/o Debiasing'] * 6 +
        # Your Model w/ Debiasing
        ['Your Model w/ Debiasing'] * 6 +
        # Your Model w/ Debiasing + Equalized Odds
        ['Your Model w/ Debiasing + Equalized Odds'] * 6 +
        # Your Model w/ Debiasing + TPR Parity
        ['Your Model w/ Debiasing + TPR Parity'] * 6 +
        # COMPAS Trained Model w/o Debiasing
        ['COMPAS Trained Model w/o Debiasing'] * 6 +
        # COMPAS Raw Model w/o Debiasing
        ['COMPAS Raw Model w/o Debiasing'] * 6,
    'Group': [
        'African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'] * 6,
    'Accuracy (%)': [
        # Your Model w/o Debiasing
        80.848806, 84.554281, 84.586466, 84.831461, 100.000000, 92.857143,
        # Your Model w/ Debiasing
        81.273210, 84.289497, 84.586466, 85.955056, 100.000000, 92.857143,
        # Your Model w/ Debiasing + Equalized Odds
        80.159151, 83.759929, 82.706767, 81.460674, 100.000000, 92.857143,
        # Your Model w/ Debiasing + TPR Parity
        80.159151, 84.995587, 84.586466, 83.146067, 100.000000, 92.857143,
        # COMPAS Trained Model w/o Debiasing
        63.477801, 62.829525, 68.796992, 60.112360, 88.888889, 85.714286,
        # COMPAS Raw Model w/o Debiasing
        63.477801, 65.641476, 65.413534, 60.674157, 100.000000, 85.714286
    ],
    'False Positive Rate': [
        # Your Model w/o Debiasing
        0.154664, 0.089530, 0.077844, 0.113208, 0.000000, 0.000000,
        # Your Model w/ Debiasing
        0.166470, 0.104704, 0.083832, 0.113208, 0.000000, 0.000000,
        # Your Model w/ Debiasing + Equalized Odds
        0.162928, 0.103187, 0.107784, 0.188679, 0.000000, 0.142857,
        # Your Model w/ Debiasing + TPR Parity
        0.161747, 0.092564, 0.083832, 0.160377, 0.000000, 0.142857,
        # COMPAS Trained Model w/o Debiasing
        0.508855, 0.309091, 0.239521, 0.198113, 0.250000, 0.142857,
        # COMPAS Raw Model w/o Debiasing
        0.403778, 0.201515, 0.191617, 0.150943, 0.000000, 0.142857
    ],
    'False Negative Rate': [
        # Your Model w/o Debiasing
        0.221580, 0.244726, 0.282828, 0.208333, 0.000000, 0.142857,
        # Your Model w/ Debiasing
        0.204239, 0.229958, 0.272727, 0.180556, 0.000000, 0.142857,
        # Your Model w/ Debiasing + Equalized Odds
        0.227360, 0.244726, 0.282828, 0.180556, 0.000000, 0.000000,
        # Your Model w/ Debiasing + TPR Parity
        0.228324, 0.229958, 0.272727, 0.180556, 0.000000, 0.000000,
        # COMPAS Trained Model w/o Debiasing
        0.248804, 0.458159, 0.434343, 0.694444, 0.000000, 0.142857,
        # COMPAS Raw Model w/o Debiasing
        0.333971, 0.539749, 0.606061, 0.750000, 0.000000, 0.142857
    ]
})

# Display the DataFrame
recidivism_group_metrics



# Filter for the 'Accuracy (%)' metric
accuracy_df = recidivism_group_metrics[['Model', 'Group', 'Accuracy (%)']]

# Pivot the DataFrame to have Models as columns
accuracy_pivot = accuracy_df.pivot(index='Group', columns='Model', values='Accuracy (%)')

# Plotting
accuracy_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Recidivism Classification Accuracy by Group')
plt.ylabel('Accuracy (%)')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Positive Rate
fpr_df = recidivism_group_metrics[['Model', 'Group', 'False Positive Rate']]
fpr_pivot = fpr_df.pivot(index='Group', columns='Model', values='False Positive Rate')

fpr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Recidivism Classification False Positive Rate by Group')
plt.ylabel('False Positive Rate')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Negative Rate
fnr_df = recidivism_group_metrics[['Model', 'Group', 'False Negative Rate']]
fnr_pivot = fnr_df.pivot(index='Group', columns='Model', values='False Negative Rate')

fnr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Recidivism Classification False Negative Rate by Group')
plt.ylabel('False Negative Rate')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Load or create the DataFrame
violence_overall_metrics = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Odds',
        'Your Model w/ Debiasing + TPR Parity',
        'COMPAS Trained Model w/o Debiasing',
        'COMPAS Raw Model w/o Debiasing'
    ],
    'Accuracy (%)': [
        94.667431,
        94.753440,
        94.151376,
        94.008028,
        92.714286,
        71.171429
    ],
    'Precision (%)': [
        94.058644,
        94.162076,
        None,
        None,
        85.959388,
        None
    ],
    'Recall (%)': [
        94.667431,
        94.753440,
        None,
        None,
        92.714286,
        None
    ],
    'F1 Score (%)': [
        93.715073,
        93.862710,
        None,
        None,
        89.209150,
        None
    ]
})

# Melt the DataFrame
metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
df_melted = violence_overall_metrics.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

# Remove rows with None values
df_melted = df_melted.dropna(subset=['Value'])

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
plt.title('Violence Classification Overall Metrics')
plt.ylabel('Percentage')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Load or create the DataFrame
# (Assuming violence_group_metrics is already created as per your data)

# Data for Violence Classification Group-specific Metrics
violence_group_metrics = pd.DataFrame({
    'Model': [
        # Your Model w/o Debiasing
        'Your Model w/o Debiasing'] * 6 +
        # Your Model w/ Debiasing
        ['Your Model w/ Debiasing'] * 6 +
        # Your Model w/ Debiasing + Equalized Odds
        ['Your Model w/ Debiasing + Equalized Odds'] * 6 +
        # Your Model w/ Debiasing + TPR Parity
        ['Your Model w/ Debiasing + TPR Parity'] * 6 +
        # COMPAS Trained Model w/o Debiasing
        ['COMPAS Trained Model w/o Debiasing'] * 6 +
        # COMPAS Raw Model w/o Debiasing
        ['COMPAS Raw Model w/o Debiasing'] * 6,
    'Group': [
        'African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'] * 6,
    'Accuracy (%)': [
        # Your Model w/o Debiasing
        94.167550, 95.590829, 95.112782, 94.413408, 100.000000, 78.571429,
        # Your Model w/ Debiasing
        94.273595, 95.679012, 95.488722, 93.854749, 100.000000, 78.571429,
        # Your Model w/ Debiasing + Equalized Odds
        93.637328, 94.797178, 95.488722, 94.972067, 100.000000, 71.428571,
        # Your Model w/ Debiasing + TPR Parity
        93.690350, 94.620811, 94.736842, 94.972067, 100.000000, 57.142857,
        # COMPAS Trained Model w/o Debiasing
        91.653460, 94.029851, 93.984962, 94.972067, 100.000000, 71.428571,
        # COMPAS Raw Model w/o Debiasing
        63.021659, 80.421422, 79.323308, 85.474860, 88.888889, 71.428571
    ],
    'False Positive Rate': [
        # Your Model w/o Debiasing
        0.009838, 0.002814, 0.000000, 0.017647, 0.000000, 0.000000,
        # Your Model w/ Debiasing
        0.009838, 0.002814, 0.000000, 0.023529, 0.000000, 0.000000,
        # Your Model w/ Debiasing + Equalized Odds
        0.013889, 0.007505, 0.000000, 0.017647, 0.000000, 0.100000,
        # Your Model w/ Debiasing + TPR Parity
        0.016782, 0.010319, 0.008000, 0.017647, 0.000000, 0.500000,
        # COMPAS Trained Model w/o Debiasing
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        # COMPAS Raw Model w/o Debiasing
        0.365418, 0.158730, 0.168000, 0.141176, 0.111111, 0.200000
    ],
    'False Negative Rate': [
        # Your Model w/o Debiasing
        0.588608, 0.691176, 0.812500, 0.777778, 0.000000, 0.750000,
        # Your Model w/ Debiasing
        0.575949, 0.676471, 0.750000, 0.777778, 0.000000, 0.750000,
        # Your Model w/ Debiasing + Equalized Odds
        0.607595, 0.750000, 0.750000, 0.666667, None,      0.750000,
        # Your Model w/ Debiasing + TPR Parity
        0.569620, 0.735294, 0.750000, 0.666667, None,      0.250000,
        # COMPAS Trained Model w/o Debiasing
        1.000000, 1.000000, 1.000000, 1.000000, None,      1.000000,
        # COMPAS Raw Model w/o Debiasing
        0.417722, 0.779412, 0.812500, 0.222222, None,      0.500000
    ]
})

# Display the DataFrame
violence_group_metrics


# Filter for the 'Accuracy (%)' metric
accuracy_df = violence_group_metrics[['Model', 'Group', 'Accuracy (%)']]

# Pivot the DataFrame
accuracy_pivot = accuracy_df.pivot(index='Group', columns='Model', values='Accuracy (%)')

# Plotting
accuracy_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Violence Classification Accuracy by Group')
plt.ylabel('Accuracy (%)')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Positive Rate
fpr_df = violence_group_metrics[['Model', 'Group', 'False Positive Rate']]
fpr_pivot = fpr_df.pivot(index='Group', columns='Model', values='False Positive Rate')

fpr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Violence Classification False Positive Rate by Group')
plt.ylabel('False Positive Rate')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Negative Rate
fnr_df = violence_group_metrics[['Model', 'Group', 'False Negative Rate']]
fnr_pivot = fnr_df.pivot(index='Group', columns='Model', values='False Negative Rate')

# Handle None values by filling with NaN
fnr_pivot = fnr_pivot.apply(pd.to_numeric, errors='coerce')

fnr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Violence Classification False Negative Rate by Group')
plt.ylabel('False Negative Rate')
plt.xlabel('Demographic Group')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Assuming df_melted from previous steps

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='Metric', y='Value', hue='Model', marker='o')
plt.title('Model Performance Across Metrics')
plt.ylabel('Percentage')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Assuming recidivism_group_metrics is already loaded

# Calculate the difference between FNR and FPR
recidivism_group_metrics['FNR - FPR'] = recidivism_group_metrics['False Negative Rate'] - recidivism_group_metrics['False Positive Rate']

# Display the updated DataFrame
recidivism_group_metrics.head()

# Pivot the DataFrame
heatmap_data = recidivism_group_metrics.pivot(index='Group', columns='Model', values='FNR - FPR')

# Sort the groups for consistent ordering (optional)
heatmap_data = heatmap_data.reindex(['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'])

# Display the pivoted DataFrame
heatmap_data

# Set up the matplotlib figure
plt.figure(figsize=(12, 6))

# Create a diverging colormap (e.g., blue for negative, red for positive differences)
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Plot the heatmap
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap, center=0, linewidths=.5, cbar_kws={'label': 'FNR - FPR'})

# Customize the plot
plt.title('Difference Between False Negative Rate and False Positive Rate\n(Recidivism Classification)')
plt.ylabel('Demographic Group')
plt.xlabel('Model')

# Adjust layout for better fit
plt.tight_layout()

# Show the plot
plt.show()
