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

# Display the DataFrame
violence_regression_overall
