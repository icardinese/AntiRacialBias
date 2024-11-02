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
