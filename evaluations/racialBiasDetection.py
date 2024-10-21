from fairlearn.metrics import MetricFrame
from sklearn.metrics import confusion_matrix
import pprint

# Calculate confusion matrix for racial subgroups
def group_confusion_matrix(y_true, y_pred, sensitive_features):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        return {
            "False Positive Rate": fp / (fp + tn),
            "False Negative Rate": fn / (fn + tp)
        }
    else:
        return {
            "Confusion Matrix": cm
        }

# Evaluate bias for each racial gruop
def evaluate_bias(X_test, y_test, y_pred, data, x_test_indices, section_equalizer):
    X_test_original = data.loc[x_test_indices]
    racial_groups = X_test_original[section_equalizer].unique()
    bias_metrics = {}
    for group in racial_groups:
        # This makes a boolean array. Basically if they are equal to the group true.
        # If not the array appends false. this tracks which values should count
        # depending on the group selected
        group_mask = X_test_original[section_equalizer] == group
        bias_metrics[group] = group_confusion_matrix(y_test[group_mask], y_pred[group_mask], X_test_original[section_equalizer][group_mask])

    pprint.pp(bias_metrics)