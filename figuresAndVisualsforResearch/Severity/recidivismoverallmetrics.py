import pandas as pd

# Data for Crime Severity Classification Overall Metrics
crime_severity_overall_metrics = pd.DataFrame({
    'Model': [
        'Your Model w/o Debiasing',
        'Your Model w/ Debiasing',
        'Your Model w/ Debiasing + Equalized Odds',
        'Your Model w/ Debiasing + TPR Parity',
        'COMPAS Trained Model w/o Debiasing'
    ],
    'Accuracy (%)': [
        75.494308,  # Your Model w/o Debiasing
        75.554224,  # Your Model w/ Debiasing
        65.368484,  # Your Model w/ Debiasing + Equalized Odds
        72.798083,  # Your Model w/ Debiasing + TPR Parity
        33.273810   # COMPAS Trained Model w/o Debiasing
    ],
    'Precision (%)': [
        69.892125,  # Your Model w/o Debiasing
        69.943896,  # Your Model w/ Debiasing
        None,       # Not provided after post-processing
        None,
        11.071464   # COMPAS Trained Model w/o Debiasing
    ],
    'Recall (%)': [
        75.494308,  # Your Model w/o Debiasing
        75.554224,  # Your Model w/ Debiasing
        None,
        None,
        33.273810   # COMPAS Trained Model w/o Debiasing
    ],
    'F1 Score (%)': [
        72.570736,  # Your Model w/o Debiasing
        72.632329,  # Your Model w/ Debiasing
        None,
        None,
        16.614613   # COMPAS Trained Model w/o Debiasing
    ]
})

# Display the DataFrame
crime_severity_overall_metrics
