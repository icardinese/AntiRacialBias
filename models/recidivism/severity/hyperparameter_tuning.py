import sys
import os
from sklearn.model_selection import RandomizedSearchCV
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Change to classifier
from xgboost import XGBClassifier  # Change to classifier

# RandomForest hyperparameter grid for classification
RandomForest_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
}

# XGBoost hyperparameter grid for multi-class classification
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'objective': ['multi:softmax'],  # Use softmax for multi-class classification
    'num_class': [9]  # Adjust this based on the number of classes
}

def get_xgb_param_grid():
    return xgb_param_grid

def get_random_forest_param_grid():
    return RandomForest_param_grid

# Function to perform hyperparameter tuning
def hyperparametertuning(model, param_grid, X_train, y_train, label):
    # Perform hyperparameter tuning using RandomizedSearchCV
    search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring='accuracy', cv=5, random_state=42)
    search.fit(X_train, y_train)

    # Get best parameters and save them to a new JSON file named 'classification_hyperparameters.json'
    best_params = search.best_params_
    try:
        with open('severity_hyperparameters.json', 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[label] = best_params
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except FileNotFoundError:
        with open('severity_hyperparameters.json', 'w') as f:
            data = {label: best_params}
            json.dump(data, f, indent=4)

    print(f"Best parameters for {label}: {best_params}")

# Example usage for multi-class classification
# hyperparametertuning(XGBClassifier(), get_xgb_param_grid(), X_train, y_train, 'xgb_model')
# hyperparametertuning(RandomForestClassifier(), get_random_forest_param_grid(), X_train, y_train, 'random_forest_model')

