from xgboost import XGBRegressor
import os
import json
from . import hyperparameter_tuning as ht

class XGBoostModel:
    def __init__(self):
        # XGBoost for regression
        self.model = XGBRegressor(objective='reg:squarederror')
        self.best_model = None
        self.y_pred = None
        self.label = 'xgb_model'

    def fit(self, x_train, y_train):
        best_params = None

        # Load or tune hyperparameters
        if os.path.exists('regression_hyperparameters.json'):
            with open('regression_hyperparameters.json', 'r') as f:
                try:
                    data = json.load(f)
                    if self.label in data:
                        best_params = data[self.label]
                except json.JSONDecodeError:
                    print("Error loading best hyperparameters.")
        
        if best_params is None:
            ht.hyperparametertuning(self.model, ht.get_xgb_param_grid(), x_train, y_train, self.label)
            if os.path.exists('regression_hyperparameters.json'):
                with open('regression_hyperparameters.json', 'r') as f:
                    try:
                        data = json.load(f)
                        if self.label in data:
                            best_params = data[self.label]
                    except json.JSONDecodeError:
                        print("Error loading best hyperparameters.")
        
        self.best_model = XGBRegressor(**best_params)
        self.best_model.fit(x_train, y_train)

    def predict(self, x_test):
        self.y_pred = self.best_model.predict(x_test)
        return self.y_pred

    def get_model(self):
        return self.best_model
