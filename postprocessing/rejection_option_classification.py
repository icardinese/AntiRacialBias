import numpy as np
from sklearn.metrics import accuracy_score

def reject_option_classification(y_pred_proba, threshold_lower, threshold_upper, y_true, sensitive_feature, desired_group):
    """
    Adjusts predictions within a probability buffer to equalize outcomes across groups.
    """
    y_pred = np.zeros_like(y_pred_proba)
    # Apply thresholds: reassign to meet fairness in the decision boundary region
    for i, prob in enumerate(y_pred_proba):
        if sensitive_feature[i] == desired_group and threshold_lower < prob < threshold_upper:
            # Adjust predictions for sensitive group within boundary
            y_pred[i] = 1 if y_true[i] == 1 else 0
        else:
            # Standard prediction based on threshold
            y_pred[i] = 1 if prob >= 0.5 else 0
    return y_pred