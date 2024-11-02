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

# Display the DataFrame
recidivism_regression_overall
