from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss
import numpy as np

# Assuming y_pred_proba are the predicted probabilities from your model for X_test
# y_test is the true labels for X_test
# X_test_original contains the sensitive feature column (e.g., 'race')

# Separate data by group for calibration
def isotonic_regression(data, X_test_indices, y_test, y_pred_proba, section_equalizer):
    X_test_original = data.loc[X_test_indices]
    groups = X_test_original[section_equalizer].unique()
    calibrated_preds = np.zeros_like(y_pred_proba)  # Initialize calibrated predictions array

    for group in groups:
        # Filter data for current group
        group_indices = X_test_original[X_test_original[section_equalizer] == group].index
        group_y_pred_proba = y_pred_proba[group_indices]
        group_y_test = y_test[group_indices]

        # Apply isotonic regression for monotonic calibration within the group
        iso_reg = IsotonicRegression(out_of_bounds='clip')  # Clip out-of-bound predictions
        iso_reg.fit(group_y_pred_proba, group_y_test)

        # Get calibrated probabilities for the group
        calibrated_preds[group_indices] = iso_reg.predict(group_y_pred_proba)

    # Evaluate accuracy and calibration score after monotonic calibration
    accuracy = accuracy_score(y_test, calibrated_preds.round())
    brier_score = brier_score_loss(y_test, calibrated_preds)

    print(f"Monotonic Calibration - Accuracy: {accuracy * 100:.2f}%")
    print(f"Monotonic Calibration - Brier Score: {brier_score:.4f}")
