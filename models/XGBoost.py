from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier()
        self.best_model = None
        self.y_pred = None

    def fit(self, x_train, y_train):
        # Hyperparameter tuning
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.3, 0.7, 1.0]
        }

        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, cv=5, scoring='accuracy', n_iter=20, random_state=42)
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")

        # Train the best model
        self.best_model = XGBClassifier(**best_params)
        self.best_model.fit(x_train, y_train)

    def predict(self, x_test):
        self.y_pred = self.best_model.predict(x_test)
        return self.y_pred

    def get_y_pred(self):
        return self.y_pred