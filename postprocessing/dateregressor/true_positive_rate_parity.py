from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

y_pred_fixed = None

# True Positive Rate Parity adapted for regression
def equalize(data, metaclassifier, X_train, y_train, X_test, y_test, x_test_indices, x_train_indices, section_equalizer, xgboost, randomforest):
    X_test_original = data.loc[x_test_indices]
    X_train_original = data.loc[x_train_indices]

    global y_pred_fixed
    y_pred_fixed = np.zeros_like(y_test, dtype=float)

    # Iterate over unique groups in section_equalizer
    racial_groups = X_train_original[section_equalizer].unique()
    for group in racial_groups:
        group_train_indices = X_train_original[X_train_original[section_equalizer] == group].index
        group_test_indices = X_test_original[X_test_original[section_equalizer] == group].index

        # Check if group has enough data to avoid sparse data issues
        if len(group_train_indices) < 5 or len(group_test_indices) < 5:
            print(f"Skipping group {group} due to sparse data.")
            continue

        # Train on the group's training set
        xgb_train_pred = xgboost.predict(X_train[group_train_indices])
        rf_train_pred = randomforest.predict(X_train[group_train_indices])
        X_meta_train = np.column_stack((xgb_train_pred, rf_train_pred))
        
        # Fit metaclassifier only on group-specific training data
        postprocess_est = ThresholdOptimizer(
            estimator=metaclassifier,
            constraints="mean_absolute_error_parity",  # Custom constraint for regression
            objective="mean_absolute_error"
        )
        postprocess_est.fit(X_meta_train, y_train[group_train_indices], sensitive_features=X_train_original[section_equalizer].loc[group_train_indices])

        # Test on the group's test set
        xgb_test_pred = xgboost.predict(X_test[group_test_indices])
        rf_test_pred = randomforest.predict(X_test[group_test_indices])
        X_meta_test = np.column_stack((xgb_test_pred, rf_test_pred))
        y_pred_fixed[group_test_indices] = postprocess_est.predict(X_meta_test, sensitive_features=X_test_original[section_equalizer].loc[group_test_indices])

    # Calculate MAE for the entire test set and each group
    overall_mae = mean_absolute_error(y_test, y_pred_fixed)
    print(f"Overall MAE after True Positive Rate Parity: {overall_mae}")
    for group in racial_groups:
        group_indices = X_test_original[X_test_original[section_equalizer] == group].index
        group_mae = mean_absolute_error(y_test[group_indices], y_pred_fixed[group_indices])
        print(f"Group {group} MAE: {group_mae}")

def get_y_pred_fixed():
    return y_pred_fixed
