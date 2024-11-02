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

# Load or create the DataFrame
crime_severity_overall_metrics = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Odds',
        'Your Model w/ Debiasing + TPR Parity',
        'COMPAS Trained Model w/o Debiasing'
    ],
    'Accuracy (%)': [
        75.494308,
        75.554224,
        65.368484,
        72.798083,
        33.273810
    ],
    'Precision (%)': [
        69.892125,
        69.943896,
        None,
        None,
        11.071464
    ],
    'Recall (%)': [
        75.494308,
        75.554224,
        None,
        None,
        33.273810
    ],
    'F1 Score (%)': [
        72.570736,
        72.632329,
        None,
        None,
        16.614613
    ]
})

# Melt the DataFrame for easier plotting
metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
df_melted = crime_severity_overall_metrics.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

# Remove rows with None values
df_melted = df_melted.dropna(subset=['Value'])

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
plt.title('Crime Severity Classification Overall Metrics')
plt.ylabel('Percentage')
plt.xlabel('Metric')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

groups = ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian']

# Accuracy per group for each model
accuracy_data = {
    'Your Model w/o Debiasing': [73.596059, 78.137652, 75.268817, 91.071429, 66.666667, 40.0],
    'Your Model w/ Debiasing': [73.694581, 78.340081, 74.193548, 91.071429, 66.666667, 40.0],
    'Your Model w/ Debiasing + Equalized Odds': [62.955665, 69.433198, 66.666667, 75.0, 66.666667, 20.0],
    'Your Model w/ Debiasing + TPR Parity': [71.822660, 74.493927, 75.268817, 76.785714, 66.666667, 20.0],
    'COMPAS Trained Model w/o Debiasing': [31.925344, 35.400000, 30.526316, 44.642857, 0.0, 60.0]
}

# False Positive Rate per group for each model
fpr_data = {
    'Your Model w/o Debiasing': [0.0, 0.0, 0.0, 0.055556, 0.0, 0.25],
    'Your Model w/ Debiasing': [0.0, 0.0, 0.0, 0.055556, 0.0, 0.25],
    'Your Model w/ Debiasing + Equalized Odds': [0.0, 0.0, 0.0, 0.125, 0.0, 0.2],
    'Your Model w/ Debiasing + TPR Parity': [0.128643, 0.109731, 0.152174, 0.142857, 0.166667, 0.4],
    'COMPAS Trained Model w/o Debiasing': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# False Negative Rate per group for each model
fnr_data = {
    'Your Model w/o Debiasing': [1.0, 1.0, 1.0, 0.1, 1.0, 0.0],
    'Your Model w/ Debiasing': [1.0, 1.0, 1.0, 0.1, 1.0, 0.0],
    'Your Model w/ Debiasing + Equalized Odds': [0.25, 0.636364, 0.0, 0.0, 1.0, 0.0],
    'Your Model w/ Debiasing + TPR Parity': [0.05, 0.181818, 0.0, 0.0, 0.0, 0.0],
    'COMPAS Trained Model w/o Debiasing': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}

# Collect data in a list of dictionaries
data = []

for model in accuracy_data.keys():
    for i, group in enumerate(groups):
        data.append({
            'Model': model,
            'Group': group,
            'Accuracy (%)': accuracy_data[model][i],
            'False Positive Rate': fpr_data[model][i],
            'False Negative Rate': fnr_data[model][i]
        })

# Create DataFrame from the list of dictionaries
crime_severity_group_metrics = pd.DataFrame(data)

# Display the DataFrame
print(crime_severity_group_metrics)

# Prepare an empty list to hold DataFrames
df_list = []

for model in accuracy_data.keys():
    temp_df = pd.DataFrame({
        'Model': [model] * len(groups),
        'Group': groups,
        'Accuracy (%)': accuracy_data[model],
        'False Positive Rate': fpr_data[model],
        'False Negative Rate': fnr_data[model]
    })
    df_list.append(temp_df)

# Concatenate all DataFrames
crime_severity_group_metrics = pd.concat(df_list, ignore_index=True)

# Display the DataFrame
crime_severity_group_metrics

# Calculate the difference between FNR and FPR
crime_severity_group_metrics['FNR - FPR'] = crime_severity_group_metrics['False Negative Rate'] - crime_severity_group_metrics['False Positive Rate']

# Pivot the DataFrame
heatmap_data = crime_severity_group_metrics.pivot(index='Group', columns='Model', values='FNR - FPR')

# Sort the groups (optional)
heatmap_data = heatmap_data.reindex(['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'])

# Plotting the Heatmap
plt.figure(figsize=(12, 6))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap, center=0, linewidths=.5, cbar_kws={'label': 'FNR - FPR'})
plt.title('Difference Between False Negative Rate and False Positive Rate\n(Crime Severity Classification)')
plt.ylabel('Demographic Group')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# Accuracy by Group
accuracy_df = crime_severity_group_metrics[['Model', 'Group', 'Accuracy (%)']]

# Pivot the DataFrame
accuracy_pivot = accuracy_df.pivot(index='Group', columns='Model', values='Accuracy (%)')

# Plotting
accuracy_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Crime Severity Classification Accuracy by Group')
plt.ylabel('Accuracy (%)')
plt.xlabel('Demographic Group')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Positive Rate
fpr_df = crime_severity_group_metrics[['Model', 'Group', 'False Positive Rate']]
fpr_pivot = fpr_df.pivot(index='Group', columns='Model', values='False Positive Rate')

fpr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Crime Severity Classification False Positive Rate by Group')
plt.ylabel('False Positive Rate')
plt.xlabel('Demographic Group')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# False Negative Rate
fnr_df = crime_severity_group_metrics[['Model', 'Group', 'False Negative Rate']]
fnr_pivot = fnr_df.pivot(index='Group', columns='Model', values='False Negative Rate')

# Ensure numeric data
fnr_pivot = fnr_pivot.apply(pd.to_numeric, errors='coerce')

fnr_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Crime Severity Classification False Negative Rate by Group')
plt.ylabel('False Negative Rate')
plt.xlabel('Demographic Group')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

