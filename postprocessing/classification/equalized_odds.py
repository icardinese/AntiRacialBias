from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

y_pred_fixed = None

# Apply threshold optimizer for Equalized Odds (FPR Fix)
def equalize(data, metaclassifier, X_train, y_train, X_test, y_test, x_test_indices, x_train_indices, section_equalizer, xgboost, randomforest):
    X_test_original = data.loc[x_test_indices]
    X_train_original = data.loc[x_train_indices]

    # Get the final predictions from the meta-classifier
    xgb_train_pred = xgboost.predict_proba(X_train)[:, 1]
    rf_train_pred = randomforest.predict_proba(X_train)[:, 1]
    X_meta_train = np.column_stack((xgb_train_pred, rf_train_pred))

    xgb_test_pred = xgboost.predict_proba(X_test)[:, 1]
    rf_test_pred = randomforest.predict_proba(X_test)[:, 1]
    X_meta_test = np.column_stack((xgb_test_pred, rf_test_pred))

    # Ensure that metaclassifier is wrapped in a KerasClassifier to be sklearn-compatible
    postprocess_est = ThresholdOptimizer(
        estimator=metaclassifier,  # This is now a KerasClassifier
        constraints="equalized_odds",  # Equalize FPR across groups
        objective="accuracy_score"
    )

    postprocess_est.fit(X_meta_train, y_train, sensitive_features=X_train_original[section_equalizer])
    global y_pred_fixed
    y_pred_fixed = postprocess_est.predict(X_meta_test, sensitive_features=X_test_original[section_equalizer])

    print(f"Accuracy after post-processing: {accuracy_score(y_test, y_pred_fixed) * 100.0}%")
    racial_groups = X_test_original[section_equalizer].unique()

    y_test_series = pd.Series(y_test, index=x_test_indices)

    for group in racial_groups:
        group_indices = X_test_original[X_test_original[section_equalizer] == group].index
        group_y_test = y_test_series.loc[group_indices]
        group_y_pred = pd.Series(y_pred_fixed, index=x_test_indices).loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")
    
def get_y_pred_fixed():
    return y_pred_fixed


