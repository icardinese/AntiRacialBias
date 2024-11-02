# Data for Recidivism Classification Group-specific Metrics
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
