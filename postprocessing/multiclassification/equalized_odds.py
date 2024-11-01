from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

y_pred_fixed = None

# Equalized Odds for Multiclass with handling for sparse groups
def equalize(data, metaclassifier, X_train, y_train, X_test, y_test, x_test_indices, x_train_indices, section_equalizer, xgboost, randomforest):
    X_test_original = data.loc[x_test_indices]
    X_train_original = data.loc[x_train_indices]

    # Placeholder for storing post-processed predictions
    global y_pred_fixed
    y_pred_fixed = np.zeros_like(y_test)

    # For each class in the multiclass setup
    for class_label in np.unique(y_train):
        # Convert multiclass problem into binary (one-vs-rest) for each class
        y_train_binary = (y_train == class_label).astype(int)
        
        # Get base model predictions on the training set for the meta-classifier
        xgb_train_pred = xgboost.predict_proba(X_train)[:, class_label]
        rf_train_pred = randomforest.predict_proba(X_train)[:, class_label]
        X_meta_train = np.column_stack((xgb_train_pred, rf_train_pred))
        
        # Check representation in the training set
        sparse_groups = X_train_original[section_equalizer].value_counts() < 5
        sparse_groups = sparse_groups[sparse_groups].index.tolist()

        # Apply ThresholdOptimizer for Equalized Odds on groups with sufficient data
        try:
            postprocess_est = ThresholdOptimizer(
                estimator=metaclassifier,
                constraints="equalized_odds",
                objective="accuracy_score"
            )
            postprocess_est.fit(
                X_meta_train, y_train_binary,
                sensitive_features=X_train_original[section_equalizer].where(~X_train_original[section_equalizer].isin(sparse_groups))
            )

            # Now get the test set predictions from the base models
            xgb_test_pred = xgboost.predict_proba(X_test)[:, class_label]
            rf_test_pred = randomforest.predict_proba(X_test)[:, class_label]
            X_meta_test = np.column_stack((xgb_test_pred, rf_test_pred))

            # Post-process the test predictions
            class_pred = postprocess_est.predict(
                X_meta_test,
                sensitive_features=X_test_original[section_equalizer].where(~X_test_original[section_equalizer].isin(sparse_groups))
            )
            y_pred_fixed[class_pred == 1] = class_label
            
        except ValueError as e:
            print(f"Skipping post-processing for class {class_label} due to sparse groups: {sparse_groups}")

    # Evaluate overall accuracy after post-processing
    overall_accuracy = accuracy_score(y_test, y_pred_fixed)
    print(f"Overall accuracy after Equalized Odds post-processing: {overall_accuracy * 100.0}%")

    # Group-wise accuracy
    racial_groups = X_test_original[section_equalizer].unique()
    y_test_series = pd.Series(y_test, index=x_test_indices)
    y_pred_series = pd.Series(y_pred_fixed, index=x_test_indices)

    for group in racial_groups:
        group_indices = X_test_original[X_test_original[section_equalizer] == group].index
        group_y_test = y_test_series.loc[group_indices]
        group_y_pred = y_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

def get_y_pred_fixed():
    return y_pred_fixed
