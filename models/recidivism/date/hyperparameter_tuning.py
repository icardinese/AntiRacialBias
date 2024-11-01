import sys
import os
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Define hyperparameter space for XGBoost (Multi-class classification)
xgb_param_grid_regression = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'objective': ['reg:squarederror']  # Updated for regression
}

def get_xgb_param_grid():
    return xgb_param_grid_regression

# Define hyperparameter space for RandomForest (Multi-class classification)
RandomForest_param_grid_regression = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error']  # Suitable criteria for regression
}

def get_rf_param_grid():
    return RandomForest_param_grid_regression

# Function to perform hyperparameter tuning
def hyperparametertuning(model, param_grid, X_train, y_train, label):
    # Perform hyperparameter tuning using RandomizedSearchCV for regression
    search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring='neg_mean_squared_error', cv=5, random_state=42)
    search.fit(X_train, y_train)

    # Get best parameters and save them to a new JSON file named 'regression_hyperparameters.json'
    best_params = search.best_params_
    try:
        with open('regression_hyperparameters.json', 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[label] = best_params
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except FileNotFoundError:
        with open('regression_hyperparameters.json', 'w') as f:
            data = {label: best_params}
            json.dump(data, f, indent=4)

    print(f"Best parameters for {label}: {best_params}")


# Example of using hyperparameter tuning
# XGBClassifier and RandomForestClassifier will handle multi-class classification
# Make sure X_train and y_train are the features and labels for multi-class severity prediction

# Example usage:
# hyperparametertuning(XGBClassifier(), get_xgb_param_grid(), X_train, y_train_severity, 'xgb_model')
# hyperparametertuning(RandomForestClassifier(), get_random_forest_param_grid(), X_train, y_train_severity, 'random_forest_model')