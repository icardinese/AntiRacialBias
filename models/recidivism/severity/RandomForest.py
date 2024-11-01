from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import os
import json
from . import hyperparameter_tuning as ht

class RandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        # Set RandomForest to handle multi-class classification
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.best_model = None
        self.y_pred = None
        self.label = 'random_forest_model'

    def fit(self, X_train, y_train):
        best_params = None

        # Load or tune hyperparameters
        if os.path.exists('severity_hyperparameters.json'):
            with open('severity_hyperparameters.json', 'r') as f:
                try:
                    data = json.load(f)
                    if self.label in data:
                        best_params = data[self.label]
                except json.JSONDecodeError:
                    print("Error loading best hyperparameters.")
        
        if best_params is None:
            ht.hyperparametertuning(self.model, ht.get_random_forest_param_grid(), X_train, y_train, self.label)
            if os.path.exists('severity_hyperparameters.json'):
                with open('severity_hyperparameters.json', 'r') as f:
                    try:
                        data = json.load(f)
                        if self.label in data:
                            best_params = data[self.label]
                    except json.JSONDecodeError:
                        print("Error loading best hyperparameters.")
        
        self.best_model = RandomForestClassifier(**best_params)
        self.best_model.fit(X_train, y_train)

    def predict(self, X_test):
        self.y_pred = self.best_model.predict(X_test)
        return self.y_pred

    def predict_proba(self, x_test):
        return self.best_model.predict_proba(x_test)

    def get_y_pred(self):
        return self.y_pred

    def get_model(self):
        return self.best_model
