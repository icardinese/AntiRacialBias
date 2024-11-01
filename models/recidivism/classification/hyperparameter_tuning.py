import sys
import os
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# # In order to access the data directory from this file
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# import data.recidivism.data as recidivism_data
# import data.violence.data as violence_data
# import data.recidivism.preproccessing as recidivism_preproccessing
# import data.violence.preproccessing as violence_preproccessing

# # Your dataset and train-test split here
# recidivismData = recidivism_data.get_data()

# # The X-Data does not change depending on classification or regression!
# X_recidivism = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
#         'juv_other_count', 'priors_count', 'days_b_screening_arrest', 
#         'c_days_from_compas','sex', 'race', 'score_text', 'decile_score']]
# # The target variable. Which is the income column
# y_recidivism_classification = recidivismData['is_recid']

# print("Unique values in y_recidivism_classification:", y_recidivism_classification.unique())

# # Convert the classification of Low, Medium, and High to 0, 1, and 2 respectively
# # ML models require numerical boolean values typically to work for classification algorithms

# # Split the dataset into training and testing sets
# X_recidivism_train, X_recidivism_test, y_recidivism_classification_train, y_recidivism_classification_test = train_test_split(
#     X_recidivism, y_recidivism_classification, test_size=0.2, random_state=42)

# # Conserves the original dataset indexes before transforming it to a csr_matrix by preprocessor
# X_recidivism_test_indices = X_recidivism_test.index
# X_recidivism_train_indices = X_recidivism_train.index

# X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)


# Example of hyperparameter space
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
def get_xgb_param_grid():
    return xgb_param_grid

RandomForest_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

def get_random_forest_param_grid():
    return RandomForest_param_grid

# Models used!
# xgb_model = XGBClassifier()
# RandomForest_model = RandomForestClassifier()

def hyperparametertuning(model, param_grid, X_train, y_train, label):
    # Hyperparameter tuning
    search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring='accuracy', cv=5, random_state=42)
    search.fit(X_train, y_train)

    # Get best parameters and save them to a file
    best_params = search.best_params_
    try:
        with open('classification_hyperparameters.json', 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[label] = best_params
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except FileNotFoundError:
        with open('classification_hyperparameters.json', 'w') as f:
            data = {label: best_params}
            json.dump(data, f, indent=4)

    print(f"Best parameters for {label}: {best_params}")

# XGB Model hyperparameter tuning. Example
# hyperparametertuning(XGBClassifier(), xgb_param_grid, X_recidivism_train, y_recidivism_classification_train, 'xgb_model')

# Random Forest hyperparameter tuning. Example
# hyperparametertuning(RandomForestClassifier(), RandomForest_param_grid, X_recidivism_train, y_recidivism_classification_train, 'random_forest_model')