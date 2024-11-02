# Data for Recidivism Date Prediction Group-specific Metrics
recidivism_regression_group = pd.DataFrame({
    'Model': [
        # Your Model w/o Debiasing
        'Your Model w/o Debiasing'] * 6 +
        # Your Model w/ Debiasing
        ['Your Model w/ Debiasing'] * 6 +
        # Your Model w/ Debiasing + Equalized Residuals
        ['Your Model w/ Debiasing + Equalized Residuals'] * 6 +
        # COMPAS Trained Model w/o Debiasing
        ['COMPAS Trained Model w/o Debiasing'] * 6,
    'Group': [
        'African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian'] * 4,
    'Mean Squared Error': [
        # Your Model w/o Debiasing
        24273.615481, 19681.148476, 21706.663301, 21111.695392, 1059.083635, 18351.555915,
        # Your Model w/ Debiasing
        24056.840134, 19446.717240, 21483.310442, 20937.501942, 841.052694, 17057.013903,
        # Your Model w/ Debiasing + Equalized Residuals
        None, None, None, None, None, None,  # MSE per group not provided after equalization
        # COMPAS Trained Model w/o Debiasing
        65395.892008, 57374.610661, 54832.956500, 51820.930448, 96559.276998, 11682.355307
    ],
    'Mean Absolute Error': [
        # Your Model w/o Debiasing
        75.283315, 62.775702, 77.763823, 78.425601, 23.379413, 97.540737,
        # Your Model w/ Debiasing
        71.808025, 59.387703, 74.948165, 75.051170, 14.809854, 92.575390,
        # Your Model w/ Debiasing + Equalized Residuals
        68.694715, 57.892316, 73.462149, 78.824292, 10.989405, 86.757004,
        # COMPAS Trained Model w/o Debiasing
        194.825919, 191.408102, 195.766588, 178.344313, 222.826843, 108.047066
    ]
})

# Display the DataFrame
recidivism_regression_group
