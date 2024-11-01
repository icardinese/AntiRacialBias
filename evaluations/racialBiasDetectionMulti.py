from sklearn.metrics import confusion_matrix
import numpy as np
import pprint

# Calculate confusion matrix for racial subgroups (Multi-class version)
def group_confusion_matrix_multiclass(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    metrics = {}

    # Calculate FPR and FNR for each class
    for i in range(num_classes):
        # True Positives (TP) for class i
        tp = cm[i, i]
        # False Negatives (FN) for class i (actual class i but predicted as something else)
        fn = np.sum(cm[i, :]) - tp
        # False Positives (FP) for class i (predicted as class i but actually something else)
        fp = np.sum(cm[:, i]) - tp
        # True Negatives (TN) for class i (everything else)
        tn = np.sum(cm) - (tp + fn + fp)

        # Calculate False Positive Rate (FPR) and False Negative Rate (FNR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics[f"Class {i}"] = {
            "False Positive Rate": fpr,
            "False Negative Rate": fnr,
            "Confusion Matrix": cm[i].tolist()
        }

    return metrics

# Evaluate bias for each racial group (Multiclass version)
def evaluate_bias(X_test, y_test, y_pred, data, x_test_indices, section_equalizer):
    X_test_original = data.loc[x_test_indices]
    racial_groups = X_test_original[section_equalizer].unique()
    bias_metrics = {}

    for group in racial_groups:
        # Filter by group
        group_mask = X_test_original[section_equalizer] == group
        # Calculate confusion matrix and bias metrics for this group
        bias_metrics[group] = group_confusion_matrix_multiclass(y_test[group_mask], y_pred[group_mask])

    pprint.pp(bias_metrics)

# Example usage:
# evaluate_bias_multiclass(X_test, y_test, y_pred, recidivismData, X_test_indices, 'race')
