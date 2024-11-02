# Data for Violence Classification Overall Metrics
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
        94.667431,  # Your Model w/o Debiasing
        94.753440,  # Your Model w/ Debiasing
        94.151376,  # Your Model w/ Debiasing + Equalized Odds
        94.008028,  # Your Model w/ Debiasing + TPR Parity
        92.714286,  # COMPAS Trained Model w/o Debiasing
        71.171429   # COMPAS Raw Model w/o Debiasing
    ],
    'Precision (%)': [
        94.058644,  # Your Model w/o Debiasing
        94.162076,  # Your Model w/ Debiasing
        None,       # Not provided after post-processing
        None,
        85.959388,  # COMPAS Trained Model w/o Debiasing
        None
    ],
    'Recall (%)': [
        94.667431,  # Your Model w/o Debiasing
        94.753440,  # Your Model w/ Debiasing
        None,
        None,
        92.714286,  # COMPAS Trained Model w/o Debiasing
        None
    ],
    'F1 Score (%)': [
        93.715073,  # Your Model w/o Debiasing
        93.862710,  # Your Model w/ Debiasing
        None,
        None,
        89.209150,  # COMPAS Trained Model w/o Debiasing
        None
    ]
})

# Display the DataFrame
violence_overall_metrics
