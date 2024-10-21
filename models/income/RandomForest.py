from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.y_pred = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def get_y_pred(self):
        return self.y_pred