from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
from . import NueralNetwork as NueralNetwork
from . import adversarial_network as AdversarialNetwork  # Added AdversarialNetwork
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import load_model
import joblib

final_pred = None

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, data=None, X_test_indices=None, section_equalizer=None, adversarial=False, training_name=None, preloadName = None):
        # Initialize the XGBoost and RandomForest models
        self.xgb_model = XGBoost.XGBoostModel()  # XGBoost model for multi-class classification
        self.rf_model = RandomForest.RandomForest()  # RandomForest model for multi-class classification
        self.adversarial = adversarial
        # Choose between standard neural network and adversarial network for meta-classification
        if self.adversarial:
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
        self.training_name = training_name  # To differentiate between different training sessions
        self.preloadName = preloadName
    
    def fit(self, race_train=None):
        # Use KFold cross-validation for stacking
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create an empty array for stacking predictions of base models
        X_meta_train = np.zeros((self.X_train.shape[0], 2))

        for train_idx, val_idx in kf.split(self.X_train):
            # Split data into training and validation sets for the current fold
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Ensure all classes are present in the training fold
            all_classes = set(range(8))  # Classes expected: 0 to 7
            missing_classes = all_classes - set(y_train_fold)
            if missing_classes:
                X_train_fold = np.concatenate([X_train_fold, np.zeros((len(missing_classes), X_train_fold.shape[1]))])
                y_train_fold = np.concatenate([y_train_fold, np.array(list(missing_classes))])

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
        self._save_models()

    def _save_models(self):
        # Save the models after training
        joblib.dump(self.xgb_model, f"xgb_model_{self.training_name}.pkl")
        joblib.dump(self.rf_model, f"rf_model_{self.training_name}.pkl")
        # Save the model
        self.meta_classifier.save(f'{self.training_name}')  # HDF5 format
        print(f"Models saved as xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and neural_network_model_{self.training_name}.h5")

    def load_models(self, preloadName=None):
        # Load the pre-trained models
        if preloadName is not None:
            self.xgb_model = joblib.load(f"xgb_model_{self.preloadName}.pkl")
            self.rf_model = joblib.load(f"rf_model_{self.preloadName}.pkl")
            self.meta_classifier = self.meta_classifier.load(f"{self.preloadName}")
            print(f"Models loaded from xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and meta_classifier_{self.training_name}.pkl")
        else:
            self.xgb_model = joblib.load(f"xgb_model_{self.preloadName}.pkl")
            self.rf_model = joblib.load(f"rf_model_{self.preloadName}.pkl")
            print(f"Models loaded from xgb_model_{self.preloadName}.pkl, rf_model_{self.preloadName}.pkl")
            self.meta_classifier = self.meta_classifier.load(f"{self.preloadName}")
            print(f"Models loaded from xgb_model_{self.preloadName}.pkl, rf_model_{self.preloadName}.pkl, and meta_classifier_{self.preloadName}.pkl")


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

        # Calculate additional metrics
        precision = precision_score(self.y_test, final_pred, average='weighted')
        recall = recall_score(self.y_test, final_pred, average='weighted')
        f1 = f1_score(self.y_test, final_pred, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, final_pred)

        print(f"Precision: {precision * 100.0}%")
        print(f"Recall: {recall * 100.0}%")
        print(f"F1 Score: {f1 * 100.0}%")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Calculate accuracy for each group defined by the section_equalizer
        X_test_original = self.data.loc[self.X_test_indices]
        racial_groups = X_test_original[self.section_equalizer].unique()

        y_test_series = pd.Series(self.y_test, index=self.X_test_indices)
        final_pred_series = pd.Series(final_pred, index=self.X_test_indices)

        for group in racial_groups:
            group_indices = X_test_original[X_test_original[self.section_equalizer] == group].index
            group_y_test = y_test_series.loc[group_indices]
            group_final_pred = final_pred_series.loc[group_indices]
            group_accuracy = accuracy_score(group_y_test, group_final_pred)
            print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

    def real_predict(self, recidivism_processed_data, violence_processed_data, recidivism_verdict, violence_verdict, recidivismPreloadName,
                     violencePreloadName):
       # Load the pre-trained models for both recidivism and violence
        
        values = []

        
        # Get recidivism out of the way first
        if recidivismPreloadName is not None:
            self.preloadName = recidivismPreloadName
            self.load_models()

        # Preprocess input data for recidivism
        if (recidivism_verdict == 1):
            # Get recidivism predictions from the XGBoost and RandomForest models
            xgb_recidivism_pred = self.xgb_model.predict(recidivism_processed_data)
            rf_recidivism_pred = self.rf_model.predict(violence_processed_data)
            
            # Stack recidivism predictions for meta-prediction
            X_meta_recidivism = np.column_stack((xgb_recidivism_pred, rf_recidivism_pred))
            recidivism_prob = self.meta_classifier.predict(X_meta_recidivism)
            recidivism_pred = np.argmax(recidivism_prob, axis=1)
            values.append(recidivism_prob)
            values.append(recidivism_pred)

        # Then violence.
        self.xgb_model = XGBoost.XGBoostModel()  # XGBoost model for multi-class classification
        self.rf_model = RandomForest.RandomForest()  # RandomForest model for multi-class classification

        # Choose between standard neural network and adversarial network for meta-classification
        if self.adversarial:
            self.meta_classifier = AdversarialNetwork.AdversarialNetwork(input_dim=2, num_classes=9)
        else:
            self.meta_classifier = NueralNetwork.NueralNetwork(input_dim=2, num_classes=9)

        # Get recidivism out of the way first
        if violencePreloadName is not None:
            self.preloadName = violencePreloadName
            self.load_models()

        # Preprocess input data for violence
        if (violence_verdict == 1):    
            # Get violence predictions from the XGBoost and RandomForest models
            xgb_violence_pred = self.xgb_model.predict(violence_processed_data)
            rf_violence_pred = self.rf_model.predict(violence_processed_data)
            
            # Stack violence predictions for meta-prediction
            X_meta_violence = np.column_stack((xgb_violence_pred, rf_violence_pred))
            violence_prob = self.meta_classifier.predict(X_meta_violence)
            violence_pred = np.argmax(violence_prob, axis=1)
            values.append(violence_prob)
            values.append(violence_pred)

        if len(values) == 0:
            return None
        if len(values) == 2:
            return values[0], values[1]
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]

    
    def get_final_pred(self):
        return final_pred

    def get_meta_classifier(self):
        return self.meta_classifier.get_model()

    def get_xgb_model(self):
        return self.xgb_model.get_model()

    def get_rf_model(self):
        return self.rf_model.get_model()
