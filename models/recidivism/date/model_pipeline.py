from . import RandomForest as RandomForest
from . import XGBoost as XGBoost
from . import NueralNetwork as NueralNetwork
from . import adversarial_network as AdversarialNetwork
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from . import recidpreprocessing as rp
from . import violencepreproccessing as vp

class CustomPipeline:
    def __init__(self, X_train, y_train, X_test, y_test, data, X_test_indices, section_equalizer, adversarial=False, training_name=None, preloadName = None):
        # Initialize the XGBoost and RandomForest models as regressors
        self.xgb_model = XGBoost.XGBoostModel()  # Use XGBRegressor
        self.rf_model = RandomForest.RandomForest()  # Use RandomForestRegressor

        # Choose between standard neural network and adversarial network for meta-regression
        if adversarial:
            self.meta_regressor = AdversarialNetwork.AdversarialNetwork(input_dim=2)
        else:
            self.meta_regressor = NueralNetwork.NueralNetwork(input_dim=2)

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

            # Train the base models (XGBoost and RandomForest) on the training fold
            self.xgb_model.fit(X_train_fold, y_train_fold)
            self.rf_model.fit(X_train_fold, y_train_fold)

            # Get the predictions on the validation fold from the base models
            xgb_pred = self.xgb_model.predict(X_val_fold)
            rf_pred = self.rf_model.predict(X_val_fold)

            # Stack the base model predictions as meta features for training the meta-regressor
            X_meta_train[val_idx] = np.column_stack((xgb_pred, rf_pred))

        if race_train is not None:
            self.meta_regressor.fit(X_meta_train, self.y_train, race_train)
        else:
            self.meta_regressor.fit(X_meta_train, self.y_train)
        # Save the trained models using joblib

        self._save_models()

    def _save_models(self):
        # Save the models after training
        joblib.dump(self.xgb_model, f"xgb_model_{self.training_name}.pkl")
        joblib.dump(self.rf_model, f"rf_model_{self.training_name}.pkl")
        # Save the model
        self.meta_regressor.save(f'{self.training_name}')  # HDF5 format
        print(f"Models saved as xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and neural_network_model_{self.training_name}.h5")

    def load_models(self, preloadName=None):
        # Load the pre-trained models
        if preloadName is not None:
            self.xgb_model = joblib.load(f"xgb_model_{self.preloadName}.pkl")
            self.rf_model = joblib.load(f"rf_model_{self.preloadName}.pkl")
            self.meta_regressor = load_model(f"neural_network_model_{self.preloadName}.h5")
            print(f"Models loaded from xgb_model_{self.preloadName}.pkl, rf_model_{self.preloadName}.pkl, and neural_network_model_{self.preloadName}.h5")
        else:
            self.xgb_model = joblib.load(f"xgb_model_{self.preloadName}.pkl")
            self.rf_model = joblib.load(f"rf_model_{self.preloadName}.pkl")
            self.meta_regressor = self.meta_regressor.load(f"{self.preloadName}")
            print(f"Models loaded from xgb_model_{self.training_name}.pkl, rf_model_{self.training_name}.pkl, and meta_classifier_{self.training_name}.pkl")
        

    def predict(self):
        # Get test set predictions from the base models (XGBoost and RandomForest)
        xgb_test_pred = self.xgb_model.predict(self.X_test)
        rf_test_pred = self.rf_model.predict(self.X_test)

        # Stack the base model predictions as meta features for testing
        X_meta_test = np.column_stack((xgb_test_pred, rf_test_pred))

        # Use the meta-regressor to predict final values
        final_pred = self.meta_regressor.predict(X_meta_test)

        # Evaluate using mean squared error
        mse = mean_squared_error(self.y_test, final_pred)
        print(f"Overall Mean Squared Error: {mse}")

        mae = mean_absolute_error(self.y_test, final_pred)
        print(f"Overall Mean Absolute Error: {mae}")
        
        # Now calculate MSE and MAE for each racial group
        self.evaluate_group_wise_mse_mae(final_pred)

        return final_pred
    
    def real_predict(self, input_data, recidivism_verdict, violence_verdict, preloadName=None):
        if preloadName is not None:
            self.load_models(preloadName)

        values = []

        # Preprocess input data for recidivism
        if (recidivism_verdict == 1):
            recidivism_data = input_data.copy()  # Copy the input so we can modify for recidivism
            recidivism_data = recidivism_data.drop(colums=['v_decile_score', 'v_score_text'])
            rp.preprocessor(recidivism_data)  # Preprocess only relevant fields for recidivism

            # Get recidivism predictions from the XGBoost and RandomForest models
            xgb_recidivism_pred = self.xgb_model.predict(recidivism_data)
            rf_recidivism_pred = self.rf_model.predict(recidivism_data)
            
            # Stack recidivism predictions for meta-prediction
            X_meta_recidivism = np.column_stack((xgb_recidivism_pred, rf_recidivism_pred))
            recidivism_pred = self.meta_regressor.predict(X_meta_recidivism)
            recidivism_prob = self.meta_regressor.predict_proba(X_meta_recidivism) 
            values.append(recidivism_pred)
            values.append(recidivism_prob)

        # Preprocess input data for violence
        if (violence_verdict == 1):
            violence_data = input_data.copy()  # Copy the input so we can modify for violence
            violence_data = violence_data.drop(columns=['decile_score', 'score_text'])
            vp.preprocessor(violence_data)  # Preprocess only relevant fields for violence
    
            # Get violence predictions from the XGBoost and RandomForest models
            xgb_violence_pred = self.xgb_model.predict(violence_data)
            rf_violence_pred = self.rf_model.predict(violence_data)
            
            # Stack violence predictions for meta-prediction
            X_meta_violence = np.column_stack((xgb_violence_pred, rf_violence_pred))
            violence_pred = self.meta_regressor.predict(X_meta_violence)
            violence_prob = self.meta_regressor.predict_proba(X_meta_violence)
            values.append(violence_pred)
            values.append(violence_prob)

        if len(values) == 0:
            return None
        if len(values) == 2:
            return values[0], values[1]
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]

    def evaluate_group_wise_mse_mae(self, final_pred):
        # Retrieve the original test data and group information
        X_test_original = self.data.loc[self.X_test_indices]
        racial_groups = X_test_original[self.section_equalizer].unique()

        # Convert predictions and ground truth to pandas Series for easy indexing
        y_test_series = pd.Series(self.y_test, index=self.X_test_indices)
        final_pred_series = pd.Series(final_pred, index=self.X_test_indices)

        # Loop through each racial group and calculate group-wise MSE and MAE
        for group in racial_groups:
            group_indices = X_test_original[X_test_original[self.section_equalizer] == group].index
            group_y_test = y_test_series.loc[group_indices]
            group_final_pred = final_pred_series.loc[group_indices]

            # Calculate MSE and MAE for the group
            group_mse = mean_squared_error(group_y_test, group_final_pred)
            group_mae = mean_absolute_error(group_y_test, group_final_pred)

            print(f"Group {group} Mean Squared Error: {group_mse}")
            print(f"Group {group} Mean Absolute Error: {group_mae}")

