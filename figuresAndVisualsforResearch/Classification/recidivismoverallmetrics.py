import pandas as pd

# Data for Recidivism Classification Overall Metrics
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
        82.639885,  # Your Model w/o Debiasing
        82.840746,  # Your Model w/ Debiasing
        81.692970,  # Your Model w/ Debiasing + Equalized Odds
        82.324247,  # Your Model w/ Debiasing + TPR Parity
        63.654561,  # COMPAS Trained Model w/o Debiasing
        64.369460   # COMPAS Raw Model w/o Debiasing
    ],
    'Precision (%)': [
        82.926242,  # Your Model w/o Debiasing
        82.980019,  # Your Model w/ Debiasing
        None,       # Not provided after post-processing
        None,
        63.787800,  # COMPAS Trained Model w/o Debiasing
        None
    ],
    'Recall (%)': [
        82.639885,  # Your Model w/o Debiasing
        82.840746,  # Your Model w/ Debiasing
        None,
        None,
        63.654561,  # COMPAS Trained Model w/o Debiasing
        None
    ],
    'F1 Score (%)': [
        82.571939,  # Your Model w/o Debiasing
        82.801245,  # Your Model w/ Debiasing
        None,
        None,
        63.639699,  # COMPAS Trained Model w/o Debiasing
        None
    ]
})

# Display the DataFrame
recidivism_overall_metrics
