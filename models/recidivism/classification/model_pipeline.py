from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
from . import NueralNetwork as NueralNetwork
from . import adversarial_network as AdversarialNetwork  # Added AdversarialNetwork
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import numpy as np

final_pred = None
final_pred_binary = None

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, y_train, X_test, y_test, data, X_test_indices, section_equalizer, adversarial=False):
        self.xgb_model = XGBoost.XGBoostModel()
        self.rf_model = RandomForest.RandomForest()

        # Choose between standard neural network and adversarial network
        if adversarial:
            self.meta_classifier = AdversarialNetwork.AdversarialNetwork(input_dim=2)
        else:
            self.meta_classifier = NueralNetwork.NueralNetwork(input_dim=2)
        
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        self.data = data
        self.X_test_indices = X_test_indices
        self.section_equalizer = section_equalizer

    def fit(self, race_train=None):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        X_meta_train = np.zeros((self.X_train.shape[0], 2))

        for train_idx, val_idx in kf.split(self.X_train):

            # Split data into training and validation sets for the current fold
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Train base models on the training fold
            self.xgb_model.fit(X_train_fold, y_train_fold)
            self.rf_model.fit(X_train_fold, y_train_fold)

            # Get the predictions on the validation fold
            xgb_pred = self.xgb_model.predict_proba(X_val_fold)[:, 1]
            rf_pred = self.rf_model.predict_proba(X_val_fold)[:, 1]

            # Stack the base model predictions as meta features
            X_meta_train[val_idx] = np.column_stack((xgb_pred, rf_pred))
        
        if race_train is not None:
            # For adversarial training, include the race data
            self.meta_classifier.fit(X_meta_train, self.y_train, race_train)
        else:
            self.meta_classifier.fit(X_meta_train, self.y_train)

    def predict(self):
        # Test set predictions
        xgb_test_pred = self.xgb_model.predict_proba(self.X_test)[:, 1]
        rf_test_pred = self.rf_model.predict_proba(self.X_test)[:, 1]
        X_meta_test = np.column_stack((xgb_test_pred, rf_test_pred))

        global final_pred
        final_pred = self.meta_classifier.predict(X_meta_test).ravel()

        global final_pred_binary
        final_pred_binary = (final_pred > 0.5).astype(int)

        # Evaluate overall accuracy
        overall_accuracy = accuracy_score(self.y_test, final_pred_binary)
        print(f"Overall accuracy: {overall_accuracy * 100.0}%")

        # Calculate group-wise accuracy
        X_test_original = self.data.loc[self.X_test_indices]
        racial_groups = X_test_original[self.section_equalizer].unique()

        y_test_series = pd.Series(self.y_test, index=self.X_test_indices)
        final_pred_series = pd.Series(final_pred_binary, index=self.X_test_indices)

        for group in racial_groups:
            group_indices = X_test_original[X_test_original[self.section_equalizer] == group].index
            group_y_test = y_test_series.loc[group_indices]
            group_y_pred = final_pred_series.loc[group_indices]
            group_accuracy = accuracy_score(group_y_test, group_y_pred)
            print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

    def get_final_pred(self):
        return final_pred

    def get_final_binary_pred(self):
        return final_pred_binary

    def get_meta_classifier(self):
        return self.meta_classifier.get_model()

    def get_xgb_model(self):
        return self.xgb_model.get_model()

    def get_rf_model(self):
        return self.rf_model.get_model()
