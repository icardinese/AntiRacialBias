# Data for Crime Severity Classification Group-specific Metrics
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

# Prepare the DataFrame
crime_severity_group_metrics = pd.DataFrame(columns=['Model', 'Group', 'Accuracy (%)', 'False Positive Rate', 'False Negative Rate'])

for model in accuracy_data.keys():
    for i, group in enumerate(groups):
        crime_severity_group_metrics = crime_severity_group_metrics.append({
            'Model': model,
            'Group': group,
            'Accuracy (%)': accuracy_data[model][i],
            'False Positive Rate': fpr_data[model][i],
            'False Negative Rate': fnr_data[model][i]
        }, ignore_index=True)

# Display the DataFrame
crime_severity_group_metrics
