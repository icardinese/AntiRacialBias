from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
from . import NueralNetwork as NueralNetwork
from . import adversarial_network as AdversarialNetwork  # Added AdversarialNetwork
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import numpy as np

final_pred = None

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, y_train, X_test, y_test, data, X_test_indices, section_equalizer, adversarial=False):
        # Initialize the XGBoost and RandomForest models
        self.xgb_model = XGBoost.XGBoostModel()  # XGBoost model for multi-class classification
        self.rf_model = RandomForest.RandomForest()  # RandomForest model for multi-class classification

        # Choose between standard neural network and adversarial network for meta-classification
        if adversarial:
            self.meta_classifier = AdversarialNetwork.AdversarialNetwork(input_dim=2, num_classes=9)
        else:
            self.meta_classifier = NueralNetwork.NueralNetwork(input_dim=2, num_classes=9)

        # Convert dataframes to NumPy arrays, if necessary
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        self.data = data
        self.X_test_indices = X_test_indices
        self.section_equalizer = section_equalizer

    def fit(self, race_train=None):
        # Use KFold cross-validation for stacking
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create an empty array for stacking predictions of base models
        X_meta_train = np.zeros((self.X_train.shape[0], 2))

        for train_idx, val_idx in kf.split(self.X_train):
            # Split data into training and validation sets for the current fold
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Train the base models (XGBoost and RandomForest) on the training fold
            self.xgb_model.fit(X_train_fold, y_train_fold)
            self.rf_model.fit(X_train_fold, y_train_fold)

            # Get the predictions on the validation fold from the base models
            xgb_pred = self.xgb_model.predict_proba(X_val_fold)
            rf_pred = self.rf_model.predict_proba(X_val_fold)

            # Stack the base model predictions as meta features for training the meta-classifier
            X_meta_train[val_idx] = np.column_stack((xgb_pred.argmax(axis=1), rf_pred.argmax(axis=1)))  # Using argmax to get predicted class

        # If race data is provided, include it for adversarial debiasing during training
        if race_train is not None:
            self.meta_classifier.fit(X_meta_train, self.y_train, race_train)
        else:
            self.meta_classifier.fit(X_meta_train, self.y_train)

    def predict(self):
        # Get test set predictions from the base models (XGBoost and RandomForest)
        xgb_test_pred = self.xgb_model.predict_proba(self.X_test)
        rf_test_pred = self.rf_model.predict_proba(self.X_test)

        # Stack the base model predictions as meta features for testing
        X_meta_test = np.column_stack((xgb_test_pred.argmax(axis=1), rf_test_pred.argmax(axis=1)))

        # Use the meta-classifier (neural network or adversarial network) to predict final classes
        global final_pred
        final_pred = self.meta_classifier.predict(X_meta_test).ravel()

        # Evaluate overall accuracy of the multi-class classification
        overall_accuracy = accuracy_score(self.y_test, final_pred)
        print(f"Overall accuracy: {overall_accuracy * 100.0}%")

        # Calculate accuracy for each group defined by the section_equalizer
        X_test_original = self.data.loc[self.X_test_indices]
        racial_groups = X_test_original[self.section_equalizer].unique()

        y_test_series = pd.Series(self.y_test, index=self.X_test_indices)
        final_pred_series = pd.Series(final_pred, index=self.X_test_indices)

        for group in racial_groups:
            group_indices = X_test_original[X_test_original[self.section_equalizer] == group].index
            group_y_test = y_test_series.loc[group_indices]
            group_y_pred = final_pred_series.loc[group_indices]
            group_accuracy = accuracy_score(group_y_test, group_y_pred)
            print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

    def get_final_pred(self):
        return final_pred

    def get_meta_classifier(self):
        return self.meta_classifier.get_model()

    def get_xgb_model(self):
        return self.xgb_model.get_model()

    def get_rf_model(self):
        return self.rf_model.get_model()
