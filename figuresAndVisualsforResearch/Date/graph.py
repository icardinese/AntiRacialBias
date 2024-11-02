import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Data for Recidivism Date Prediction Overall Metrics
recidivism_regression_overall = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Residuals',
        'COMPAS Trained Model w/o Debiasing'
    ],
    'Mean Squared Error': [
        22518.548404,  # Your Model w/o Debiasing
        22296.358281,  # Your Model w/ Debiasing
        22690.503320,  # Your Model w/ Debiasing + Equalized Residuals
        61868.867767   # COMPAS Trained Model w/o Debiasing
    ],
    'Mean Absolute Error': [
        71.675507,     # Your Model w/o Debiasing
        68.251684,     # Your Model w/ Debiasing
        65.974425,     # Your Model w/ Debiasing + Equalized Residuals
        193.288899     # COMPAS Trained Model w/o Debiasing
    ]
})

# Melt the DataFrame for easier plotting
metrics = ['Mean Squared Error', 'Mean Absolute Error']
df_melted = recidivism_regression_overall.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
plt.title('Recidivism Date Prediction Overall Metrics')
plt.ylabel('Error Value')
plt.xlabel('Metric')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Data for Recidivism Date Prediction Group-specific Metrics
# We need to handle the None values appropriately

recidivism_regression_group = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing'] * 6 +
        ['Your Model w/ Debiasing'] * 6 +
        ['Your Model w/ Debiasing + Equalized Residuals'] * 6 +
        ['COMPAS Trained Model w/o Debiasing'] * 6,
    'Group': ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'] * 4,
    'Mean Squared Error': [
        24273.615481, 19681.148476, 21706.663301, 21111.695392, 1059.083635, 18351.555915,
        24056.840134, 19446.717240, 21483.310442, 20937.501942, 841.052694, 17057.013903,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  # MSE per group not provided after equalization
        65395.892008, 57374.610661, 54832.956500, 51820.930448, 96559.276998, 11682.355307
    ],
    'Mean Absolute Error': [
        75.283315, 62.775702, 77.763823, 78.425601, 23.379413, 97.540737,
        71.808025, 59.387703, 74.948165, 75.051170, 14.809854, 92.575390,
        68.694715, 57.892316, 73.462149, 78.824292, 10.989405, 86.757004,
        194.825919, 191.408102, 195.766588, 178.344313, 222.826843, 108.047066
    ]
})

# Display the DataFrame
print(recidivism_regression_group)

# Filter for Mean Absolute Error
mae_df = recidivism_regression_group[['Model', 'Group', 'Mean Absolute Error']]

# Pivot the DataFrame
mae_pivot = mae_df.pivot(index='Group', columns='Model', values='Mean Absolute Error')

# Plotting
mae_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Recidivism Date Prediction Mean Absolute Error by Group')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Demographic Group')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Attempting to save the plot again for download with an alternative approach

plt.savefig("False_Positive_Rate_Comparison.png", format="png", dpi=300)


# Data for Violence Date Prediction Overall Metrics
violence_regression_overall = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Residuals',
        'COMPAS Trained Model w/o Debiasing'
    ],
    'Mean Squared Error': [
        43799.017285,  # Your Model w/o Debiasing
        43227.257521,  # Your Model w/ Debiasing
        43594.264458,  # Your Model w/ Debiasing + Equalized Residuals
        85701.785687   # COMPAS Trained Model w/o Debiasing
    ],
    'Mean Absolute Error': [
        143.635831,    # Your Model w/o Debiasing
        143.902196,    # Your Model w/ Debiasing
        129.209173,    # Your Model w/ Debiasing + Equalized Residuals
        238.208682     # COMPAS Trained Model w/o Debiasing
    ]
})

# Melt the DataFrame
metrics = ['Mean Squared Error', 'Mean Absolute Error']
df_melted = violence_regression_overall.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model')
plt.title('Violence Date Prediction Overall Metrics')
plt.ylabel('Error Value')
plt.xlabel('Metric')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import numpy as np

# Data for Violence Date Prediction Group-specific Metrics
violence_regression_group = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing'] * 6 +
        ['Your Model w/ Debiasing'] * 6 +
        ['Your Model w/ Debiasing + Equalized Residuals'] * 6 +
        ['COMPAS Trained Model w/o Debiasing'] * 6,
    'Group': ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'] * 4,
    'Mean Squared Error': [
        41923.283354, 45816.397152, 65395.098014, 34663.040917, 97.358679, 1850.453216,
        41722.210195, 44595.551107, 62527.762016, 36290.308756, 104.637984, 2571.656155,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  # MSE per group not provided after equalization
        90003.170796, 59925.825011, 159324.930066, 78465.499696, 103789.152257, 20483.786055
    ],
    'Mean Absolute Error': [
        140.529433, 147.603314, 177.376989, 139.019660, 9.867050, 43.011322,
        141.230413, 146.867616, 173.912691, 144.200048, 10.229271, 50.147942,
        126.785672, 132.217496, 153.003005, 123.969627, 121.412659, 37.288651,
        249.081972, 197.440041, 309.996936, 238.358956, 322.163239, 115.860855
    ]
})

# Display the DataFrame
print(violence_regression_group)

# Filter for Mean Absolute Error
mae_df = violence_regression_group[['Model', 'Group', 'Mean Absolute Error']]

# Pivot the DataFrame
mae_pivot = mae_df.pivot(index='Group', columns='Model', values='Mean Absolute Error')

# Plotting
mae_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Violence Date Prediction Mean Absolute Error by Group')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Demographic Group')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Pivot the DataFrame
heatmap_data_recidivism = recidivism_regression_group.pivot(index='Group', columns='Model', values='Mean Absolute Error')

# Plotting the Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_recidivism, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'Mean Absolute Error'})
plt.title('Mean Absolute Error by Group and Model\n(Recidivism Date Prediction)')
plt.ylabel('Demographic Group')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# Pivot the DataFrame
heatmap_data_violence = violence_regression_group.pivot(index='Group', columns='Model', values='Mean Absolute Error')

# Plotting the Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_violence, annot=True, fmt=".2f", cmap='YlOrRd', cbar_kws={'label': 'Mean Absolute Error'})
plt.title('Mean Absolute Error by Group and Model\n(Violence Date Prediction)')
plt.ylabel('Demographic Group')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

