from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
from . import NueralNetwork as NueralNetwork
from . import adversarial_network as AdversarialNetwork  # Added AdversarialNetwork
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import numpy as np
import joblib  # Add joblib for saving/loading models
from tensorflow.keras.models import load_model
import pickle
import os
import sys
from . import recidpreprocessing as rp
from . import violencepreproccessing as vp

final_pred = None
final_pred_binary = None

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, y_train, X_test, y_test, data, X_test_indices, section_equalizer, adversarial=False, training_name=None, preloadName = None):
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
        self.training_name = training_name  # To differentiate between different training sessions
        self.preloadName = preloadName

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

        # Save the trained models using joblib
        self._save_models()

    def _save_models(self):
        # Save the models after training
        joblib.dump(self.xgb_model, f"xgb_model_{self.training_name}.pkl")
        joblib.dump(self.rf_model, f"rf_model_{self.training_name}.pkl")
        # Save the model
        self.meta_classifier.save(f'{self.training_name}')  # HDF5 format
        print(f"Models saved as xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and neural_network_model_{self.training_name}.h5")

    def load_models(self):
        # Load the pre-trained models
        self.xgb_model = joblib.load(f"xgb_model_{self.preloadName}.pkl")
        self.rf_model = joblib.load(f"rf_model_{self.preloadName}.pkl")
        self.meta_classifier = self.meta_classifier.load(f"{self.preloadName}")
        print(f"Models loaded from xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and meta_classifier_{self.training_name}.pkl")

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

        # Calculate additional metrics
        precision = precision_score(self.y_test, final_pred_binary, average='weighted')
        recall = recall_score(self.y_test, final_pred_binary, average='weighted')
        f1 = f1_score(self.y_test, final_pred_binary, average='weighted')

        print(f"Precision: {precision * 100.0}%")
        print(f"Recall: {recall * 100.0}%")
        print(f"F1 Score: {f1 * 100.0}%")

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

    def real_predict(self, input_data, preloadName=None, type = None):
        # Load the pre-trained models for both recidivism and violence
        if preloadName is not None:
            self.load_models(preloadName)
        # Preprocess input data for recidivism
        if type == 'recidivism':
            recidivism_data = input_data.copy()  # Copy the input so we can modify for recidivism
            recidivism_data = recidivism_data.drop(colums=['v_decile_score', 'v_score_text'])
            rp.preprocessor(recidivism_data)  # Preprocess only relevant fields for recidivism

            # Get recidivism predictions from the XGBoost and RandomForest models
            xgb_recidivism_pred = self.xgb_model.predict(recidivism_data)
            rf_recidivism_pred = self.rf_model.predict(recidivism_data)
            # Stack recidivism predictions for meta-prediction
            X_meta_recidivism = np.column_stack((xgb_recidivism_pred, rf_recidivism_pred))
            recidivism_pred = self.meta_classifier.predict(X_meta_recidivism)
            recidivism_prob = self.meta_classifier.predict_proba(X_meta_recidivism) 
            return recidivism_pred, recidivism_prob
        
        if type == 'violence':
            # Preprocess input data for violence
            violence_data = input_data.copy()  # Copy the input so we can modify for violence
            violence_data = violence_data.drop(columns=['decile_score', 'score_text'])
            vp.preprocessor(violence_data)  # Preprocess only relevant fields for violence
            # Get violence predictions from the XGBoost and RandomForest models
            xgb_violence_pred = self.xgb_model.predict(violence_data)
            rf_violence_pred = self.rf_model.predict(violence_data)
            
            # Stack violence predictions for meta-prediction
            X_meta_violence = np.column_stack((xgb_violence_pred, rf_violence_pred))
            violence_pred = self.meta_classifier.predict(X_meta_violence)
            violence_prob = self.meta_classifier.predict_proba(X_meta_violence)

            # Generate the final predictions for both recidivism and violence
            return violence_pred, violence_prob
        
        
        

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
