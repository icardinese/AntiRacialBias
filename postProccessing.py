from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score
import pandas as pd

y_pred_fixed = None

# Apply threshold optimizer for Equalized Odds (FPR Fix)
def equalize(data, model, X_train, y_train, X_test, y_test, x_test_indices, x_train_indices):
    X_test_original = data.loc[x_test_indices]
    X_train_original = data.loc[x_train_indices]
    postprocess_est = ThresholdOptimizer(
        estimator=model,  # Original XGBoost model
        constraints="equalized_odds",  # Equalize FPR across groups
        objective="accuracy_score"
    )

    postprocess_est.fit(X_train, y_train, sensitive_features=X_train_original['race'])
    global y_pred_fixed
    y_pred_fixed = postprocess_est.predict(X_test, sensitive_features=X_test_original['race'])
    print(f"Accuracy after post-processing:, {accuracy_score(y_test, y_pred_fixed)*100.0}%")
    racial_groups = X_test_original['race'].unique()

    for group in racial_groups:
        group_indices = X_test_original[X_test_original['race'] == group].index
        group_y_test = y_test.loc[group_indices]
        group_y_pred = pd.Series(y_pred_fixed, index=y_test.index).loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

def get_y_pred_fixed():
    return y_pred_fixed