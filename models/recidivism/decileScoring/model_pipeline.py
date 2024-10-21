from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
import pandas as pd
from sklearn.metrics import accuracy_score

final_pred = None

class CustomPipeline:
    def __init__(self):
        self.xgb_model = XGBoost.XGBoostModel()
        self.rf_model = RandomForest.RandomForest()

    def fit(self, X_train, y_train, X_test, y_test, data, X_test_indices):
        self.xgb_model.fit(X_train, y_train)
        # Get the residuals from XGBoost model
        residuals = y_test - self.xgb_model.predict(X_test)
        self.rf_model.fit(X_test, residuals)

        global final_pred
        final_pred = self.xgb_model.predict(X_test) + self.rf_model.predict(X_test)

        overall_accuracy = accuracy_score(y_test, final_pred)
        print(f"Overall accuracy: {overall_accuracy * 100.0}%")

        # Calcuate gruop-wise accuracy
        X_test_original = data.loc[X_test_indices]
        racial_groups = X_test_original['race'].unique()

        for group in racial_groups:
            group_indices = X_test_original[X_test_original['race'] == group].index
            group_y_test = y_test.loc[group_indices]
            group_y_pred = pd.Series(final_pred, index=y_test.index).loc[group_indices]
            group_accuracy = accuracy_score(group_y_test, group_y_pred)
    
    def predict(self, X_test):
        xgb_pred = self.xgb_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        return xgb_pred + rf_pred
    
def get_final_pred():
    return final_pred
