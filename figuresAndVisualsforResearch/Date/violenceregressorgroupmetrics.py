# Data for Violence Date Prediction Group-specific Metrics
violence_regression_group = pd.DataFrame({
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
        41923.283354, 45816.397152, 65395.098014, 34663.040917, 97.358679, 1850.453216,
        # Your Model w/ Debiasing
        41722.210195, 44595.551107, 62527.762016, 36290.308756, 104.637984, 2571.656155,
        # Your Model w/ Debiasing + Equalized Residuals
        None, None, None, None, None, None,  # MSE per group not provided after equalization
        # COMPAS Trained Model w/o Debiasing
        90003.170796, 59925.825011, 159324.930066, 78465.499696, 103789.152257, 20483.786055
    ],
    'Mean Absolute Error': [
        # Your Model w/o Debiasing
        140.529433, 147.603314, 177.376989, 139.019660, 9.867050, 43.011322,
        # Your Model w/ Debiasing
        141.230413, 146.867616, 173.912691, 144.200048, 10.229271, 50.147942,
        # Your Model w/ Debiasing + Equalized Residuals
        126.785672, 132.217496, 153.003005, 123.969627, 121.412659, 37.288651,
        # COMPAS Trained Model w/o Debiasing
        249.081972, 197.440041, 309.996936, 238.358956, 322.163239, 115.860855
    ]
})

# Display the DataFrame
violence_regression_group
