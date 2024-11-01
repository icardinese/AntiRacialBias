import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import data.recidivism.data as recidivism_data
import data.recidivism.compasspreproccessing as recidivism_compass_preproccessing  # Import your COMPAS preprocessor
import data.violence.compasspreproccessing as violence_compass_preproccessing  # Import your COMPAS preprocessor
import data.recidivism.preproccessing as recidivism_preproccessing
import data.violence.preproccessing as violence_preproccessing
import models.recidivism.classification.model_pipeline as classification_model_pipeline
import evaluations.racialBiasDetection as racialBiasDetection
import postprocessing.classification.equalized_odds as equalized_odds
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
# <<---------------------------------------------------------------------------------------------------->>
# This is for violence + recidivism classification!



# Load recidivism data
recidivismData = recidivism_data.get_data()

# Define recidivism X and y
X_recidivism = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'score_text', 'decile_score',
                               'c_charge_degree']]
X_violence = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'v_score_text', 'v_decile_score',
                               'c_charge_degree']]
y_recidivism_classification = recidivismData['is_recid']
y_violence_classification = recidivismData['is_violent_recid']

# Split data for recidivism and violence classification
X_recidivism_train, X_recidivism_test, y_recidivism_classification_train, y_recidivism_classification_test = train_test_split(
    X_recidivism, y_recidivism_classification, test_size=0.2, random_state=42)
X_violence_train, X_violence_test, y_violence_classification_train, y_violence_classification_test = train_test_split(
    X_violence, y_violence_classification, test_size=0.2, random_state=42)

# Conserves the original dataset indexes before transforming it
X_recidivism_test_indices = X_recidivism_test.index
X_recidivism_train_indices = X_recidivism_train.index
X_violence_test_indices = X_violence_test.index
X_violence_train_indices = X_violence_train.index

# Preprocess the data for the original recidivism pipeline
X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)
X_violence_train, X_violence_test = violence_preproccessing.preprocessor(X_violence_train, X_violence_test)

# Convert to DataFrame to drop rows with NaN values
X_recidivism_train_df = pd.DataFrame(X_recidivism_train, index=X_recidivism_train_indices)
X_recidivism_test_df = pd.DataFrame(X_recidivism_test, index=X_recidivism_test_indices)
X_violence_train_df = pd.DataFrame(X_violence_train, index=X_violence_train_indices)
X_violence_test_df = pd.DataFrame(X_violence_test, index=X_violence_test_indices)

# Drop rows with NaN values
X_recidivism_train_df.dropna(inplace=True)
X_recidivism_test_df.dropna(inplace=True)
X_violence_train_df.dropna(inplace=True)
X_violence_test_df.dropna(inplace=True)

# Update indices
X_recidivism_train_indices = X_recidivism_train_df.index
X_recidivism_test_indices = X_recidivism_test_df.index
X_violence_train_indices = X_violence_train_df.index
X_violence_test_indices = X_violence_test_df.index

X_recidivism_train = X_recidivism_train_df.values
X_recidivism_test = X_recidivism_test_df.values
X_violence_train = X_violence_train_df.values
X_violence_test = X_violence_test_df.values

# Drop corresponding rows in y data
y_recidivism_classification_train = y_recidivism_classification_train.loc[X_recidivism_train_indices]
y_recidivism_classification_test = y_recidivism_classification_test.loc[X_recidivism_test_indices]
y_violence_classification_train = y_violence_classification_train.loc[X_violence_train_indices]
y_violence_classification_test = y_violence_classification_test.loc[X_violence_test_indices]



# Label encode the 'race' column for adversarial debiasing
label_encoder = LabelEncoder()

# Recidivism 
race_train = recidivismData['race'].loc[X_recidivism_train_indices].values
race_recidivism_train_encoded = label_encoder.fit_transform(race_train)

# Violence
race_train = recidivismData['race'].loc[X_violence_train_indices].values
race_violence_train_encoded = label_encoder.fit_transform(race_train)

# Train recidivism pipeline without adversarial debiasing

print("Recidivism prediction without adversarial debiasing")
pipeline = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_classification_train, X_recidivism_test, 
             y_recidivism_classification_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=False, training_name='noAdversarialNoPostRecidivismClassification')
pipeline.fit()
pipeline.predict()

# Evaluate bias for each racial group using false positives and false negatives
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_classification_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

print("Violence prediction without adversarial debiasing")
# Violence classification pipeline without adversarial debiasing
pipelineviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_classification_train, X_violence_test, 
             y_violence_classification_test, recidivismData, X_violence_test_indices, 'race', adversarial=False, training_name='noAdversarialNoPostViolenceClassification')
pipelineviolence.fit()
pipelineviolence.predict()

# Evaluate bias for violence
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_classification_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

print("Recedivism prediction with adversarial debiasing")
# Recidivism pipeline with adversarial debiasing
pipelineadversarial = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_classification_train, X_recidivism_test, 
             y_recidivism_classification_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=True, training_name='adversarialNoPostRecidivismClassification')
pipelineadversarial.fit(race_train=race_recidivism_train_encoded)
pipelineadversarial.predict()

# Evaluate bias for each racial group using false positives and false negatives (adversarial)
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_classification_test, pipelineadversarial.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

print("Violence prediction with adversarial debiasing")
# Violence classification with adversarial debiasing
pipelineadversarialviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_classification_train, X_violence_test, 
             y_violence_classification_test, recidivismData, X_violence_test_indices, 'race', adversarial=True, training_name='adversarialNoPostViolenceClassification')
pipelineadversarialviolence.fit(race_train=race_violence_train_encoded)
pipelineadversarialviolence.predict()

# Evaluate bias for violence (adversarial)
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_classification_test, pipelineadversarialviolence.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# # <<----------------------- COMPAS Implementation for Comparison ---------------------->>

# Use only the decile_score and score_text for COMPAS processing
X_compas_recidivism = recidivismData[['decile_score', 'score_text']]

# Split data for recidivism classification
X_compas_violence = recidivismData[['v_decile_score', 'v_score_text']]
y_recidivism_classification = recidivismData['is_recid']
y_violence_classification = recidivismData['is_violent_recid']

# Split data for recidivism classification
X_compas_recidivism_train, X_compas_recidivism_test, y_compas_recidivism_classification_train, y_compas_recidivism_classification_test = train_test_split(
    X_compas_recidivism, y_recidivism_classification, test_size=0.2, random_state=42)

# Split data for violence classification
X_compas_violence_train, X_compas_violence_test, y_compas_violence_classification_train, y_compas_violence_classification_test = train_test_split(
    X_compas_violence, y_violence_classification, test_size=0.2, random_state=42)

X_compas_violence_test_indices = X_compas_violence_test.index
X_compas_violence_train_indices = X_compas_violence_train.index
X_compas_recidivism_test_indices = X_compas_recidivism_test.index
X_compas_recidivism_train_indices = X_compas_recidivism_train.index

# Preprocess the COMPAS data
X_compas_recidivism_train, X_compas_recidivism_test = recidivism_compass_preproccessing.preprocessor(X_compas_recidivism_train, X_compas_recidivism_test)
X_compas_violence_train, X_compas_violence_test = violence_compass_preproccessing.preprocessor(X_compas_violence_train, X_compas_violence_test)

# Convert to DataFrame to drop rows with NaN values
X_compas_recidivism_train_df = pd.DataFrame(X_compas_recidivism_train, index=X_compas_recidivism_train_indices)
X_compas_recidivism_test_df = pd.DataFrame(X_compas_recidivism_test, index=X_compas_recidivism_test_indices)
X_compas_violence_train_df = pd.DataFrame(X_compas_violence_train, index=X_compas_violence_train_indices)
X_compas_violence_test_df = pd.DataFrame(X_compas_violence_test, index=X_compas_violence_test_indices)

# Drop rows with NaN values
X_compas_recidivism_train_df.dropna(inplace=True)
X_compas_recidivism_test_df.dropna(inplace=True)
X_compas_violence_train_df.dropna(inplace=True)
X_compas_violence_test_df.dropna(inplace=True)

# Update indices
X_compas_recidivism_train_indices = X_compas_recidivism_train_df.index
X_compas_recidivism_test_indices = X_compas_recidivism_test_df.index
X_compas_violence_train_indices = X_compas_violence_train_df.index
X_compas_violence_test_indices = X_compas_violence_test_df.index

X_compas_recidivism_train = X_compas_recidivism_train_df.values
X_compas_recidivism_test = X_compas_recidivism_test_df.values
X_compas_violence_train = X_compas_violence_train_df.values
X_compas_violence_test = X_compas_violence_test_df.values

# Drop corresponding rows in y data
y_compas_recidivism_classification_train = y_compas_recidivism_classification_train.loc[X_compas_recidivism_train_indices]
y_compas_recidivism_classification_test = y_compas_recidivism_classification_test.loc[X_compas_recidivism_test_indices]
y_compas_violence_classification_train = y_compas_violence_classification_train.loc[X_compas_violence_train_indices]
y_compas_violence_classification_test = y_compas_violence_classification_test.loc[X_compas_violence_test_indices]


# Recidivism classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_recidivism = classification_model_pipeline.CustomPipeline(X_compas_recidivism_train, y_compas_recidivism_classification_train, 
    X_compas_recidivism_test, y_compas_recidivism_classification_test, recidivismData, X_compas_recidivism_test_indices, 'race', adversarial=False, training_name='noAdversarialNoPostTrainedCompasRecidivismClassification')

print("Compas Recedivism prediction without adversarial debiasing")
pipeline_compas_recidivism.fit()
pipeline_compas_recidivism.predict()

# Evaluate bias for each racial group using COMPAS data for recidivism
racialBiasDetection.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_classification_test, 
    pipeline_compas_recidivism.get_final_binary_pred(), recidivismData, X_compas_recidivism_test_indices, 'race')

# Violence classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_violence = classification_model_pipeline.CustomPipeline(X_compas_violence_train, y_compas_violence_classification_train, 
    X_compas_violence_test, y_compas_violence_classification_test, recidivismData, X_compas_violence_test_indices, 'race', adversarial=False,
    training_name='noAdversarialNoPostTrainedCompasViolenceClassification')

print("Compas Violence prediction with adversarial debiasing")
pipeline_compas_violence.fit()
pipeline_compas_violence.predict()

# Evaluate bias for each racial group using COMPAS data for violence
racialBiasDetection.evaluate_bias(X_compas_violence_test, y_compas_violence_classification_test, 
    pipeline_compas_violence.get_final_binary_pred(), recidivismData, X_compas_violence_test_indices, 'race')

# Recidivism classification based on pure decile score
X_compas_recidivism_test_original = recidivismData.loc[X_compas_recidivism_test_indices]
recidivism_racial_groups = X_compas_recidivism_test_original['race'].unique()

final_compass_recidivism_binary = (X_compas_recidivism_test_original['decile_score'] > 5).astype(int)
y_compass_recidivism_test_series = pd.Series(y_compas_recidivism_classification_test, index=X_compas_recidivism_test_indices)
final_compass_recidivism_pred_series = pd.Series(final_compass_recidivism_binary, index=X_compas_recidivism_test_indices)

overall_accuracy = accuracy_score(y_compas_recidivism_classification_test, final_compass_recidivism_binary)
print(f"Overall accuracy: {overall_accuracy * 100.0}%")

for group in recidivism_racial_groups:
        group_indices = X_compas_recidivism_test_original[X_compas_recidivism_test_original['race'] == group].index
        group_y_test = y_compass_recidivism_test_series.loc[group_indices]
        group_y_pred = final_compass_recidivism_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

racialBiasDetection.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_classification_test, 
    final_compass_recidivism_binary, recidivismData, X_compas_recidivism_test_indices, 'race')

X_compas_violence_test_original = recidivismData.loc[X_compas_violence_test_indices]
violence_racial_groups = X_compas_violence_test_original['race'].unique()

final_compass_violence_binary = (recidivismData.loc[X_compas_violence_test_indices]['v_decile_score'] > 5).astype(int)
y_compass_violence_test_series = pd.Series(y_compas_violence_classification_test, index=X_compas_violence_test_indices)
final_compass_violence_pred_series = pd.Series(final_compass_violence_binary, index=X_compas_violence_test_indices)

overall_accuracy = accuracy_score(y_compas_violence_classification_test, final_compass_violence_binary)
print(f"Overall accuracy: {overall_accuracy * 100.0}%")

for group in violence_racial_groups:
        group_indices = X_compas_violence_test_original[X_compas_violence_test_original['race'] == group].index
        group_y_test = y_compass_violence_test_series.loc[group_indices]
        group_y_pred = final_compass_violence_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

racialBiasDetection.evaluate_bias(X_compas_violence_test, y_compas_violence_classification_test, 
    final_compass_violence_binary, recidivismData, X_compas_violence_test_indices, 'race')

# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>

# This is for classification of the severity of the Violence + Recidivism crime.
import models.recidivism.severity.model_pipeline as severity_model_pipeline
import evaluations.racialBiasDetectionMulti as racialBiasDetectionMulti

X_recidivism = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'score_text', 'decile_score',
                               'c_charge_degree']]
X_violence = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'v_score_text', 'v_decile_score',
                               'c_charge_degree']]

y_recidivism_severity = recidivismData['r_charge_degree']
y_violence_severity = recidivismData['vr_charge_degree']

# Define the static severity mapping without CO3 (since we are removing it)
severity_mapping = {'(F1)': 7, '(F2)': 6, '(F3)': 5, '(F6)': 4, '(F7)': 3, '(M1)': 2, '(M2)': 1, '(MO3)': 0}


# Filter out 'CO3' rows from both datasets
y_recidivism_severity = y_recidivism_severity[y_recidivism_severity != '(CO3)']
y_violence_severity = y_violence_severity[y_violence_severity != '(CO3)']

# Ensure X has matching indices after filtering out CO3 from y
X_recidivism = X_recidivism.loc[y_recidivism_severity.index]
X_violence = X_violence.loc[y_violence_severity.index]

# Apply the severity mapping to the filtered data
y_recidivism_severity = y_recidivism_severity.map(severity_mapping)
y_violence_severity = y_violence_severity.map(severity_mapping)

# Drop NaN values in case of any missing mappings
y_recidivism_severity = y_recidivism_severity.dropna().astype(int)
y_violence_severity = y_violence_severity.dropna().astype(int)

X_recidivism = X_recidivism.loc[y_recidivism_severity.index]
X_violence = X_violence.loc[y_violence_severity.index]

# Continue with your XGBoost model training
# y_recidivism_severity and y_violence_severity now contain the integer values [0, 1, 2, ..., 8]
# Map the target values (severity labels) using the predefined severity mapping

# Split data for recidivism and violence classification
X_recidivism_train, X_recidivism_test, y_recidivism_severity_train, y_recidivism_severity_test = train_test_split(
    X_recidivism, y_recidivism_severity, test_size=0.2, random_state=42)
X_violence_train, X_violence_test, y_violence_severity_train, y_violence_severity_test = train_test_split(
    X_violence, y_violence_severity, test_size=0.2, random_state=42)

# <<<<<<<<<<<< XGBoost missing classification value handling >>>>>>>>>>>>>>>>>>

# Now, handle missing classes only in the training set.
# Ensure all classes are present in the target variable for XGBoost

all_classes_recidivism = set(range(8))  # Classes expected: 0 to 6
all_classes_violence = set(range(8))  # Same for violence data

# Identify present classes in the training data
present_classes_recidivism = set(y_recidivism_severity_train.unique())
present_classes_violence = set(y_violence_severity_train.unique())

# Find missing classes
missing_classes_recidivism = all_classes_recidivism - present_classes_recidivism
missing_classes_violence = all_classes_violence - present_classes_violence

# If there are missing classes, artificially add them to the training set
if missing_classes_recidivism:
    # Add "fake" data points for the missing recidivism classes
    fake_X_recidivism = [X_recidivism_train[0]] * len(missing_classes_recidivism)  # Duplicate the first row
    fake_y_recidivism = list(missing_classes_recidivism)
    
    # Add to the training set
    X_recidivism_train = pd.concat([pd.DataFrame(X_recidivism_train), pd.DataFrame(fake_X_recidivism)], ignore_index=True)
    y_recidivism_severity_train = pd.concat([pd.Series(y_recidivism_severity_train), pd.Series(fake_y_recidivism)], ignore_index=True)

if missing_classes_violence:
    # Add "fake" data points for the missing violence classes
    fake_X_violence = [X_violence_train[0]] * len(missing_classes_violence)  # Duplicate the first row
    fake_y_violence = list(missing_classes_violence)
    
    # Add to the training set
    X_violence_train = pd.concat([pd.DataFrame(X_violence_train), pd.DataFrame(fake_X_violence)], ignore_index=True)
    y_violence_severity_train = pd.concat([pd.Series(y_violence_severity_train), pd.Series(fake_y_violence)], ignore_index=True)

# <<<<<<<<<<< END OF XGBoost classification handling! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Conserves the original dataset indexes before transforming it
X_recidivism_test_indices = X_recidivism_test.index
X_recidivism_train_indices = X_recidivism_train.index
X_violence_test_indices = X_violence_test.index
X_violence_train_indices = X_violence_train.index

y_recidivism_severity_test_indices = y_recidivism_severity_test.index
y_recidivism_severity_train_indices = y_recidivism_severity_train.index
y_violence_severity_test_indices = y_violence_severity_test.index
y_violence_severity_train_indices = y_violence_severity_train.index

# Preprocess the data for the original recidivism pipeline
X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)
X_violence_train, X_violence_test = violence_preproccessing.preprocessor(X_violence_train, X_violence_test)

# Convert to DataFrame to drop rows with NaN values
X_recidivism_train_df = pd.DataFrame(X_recidivism_train, index=X_recidivism_train_indices)
X_recidivism_test_df = pd.DataFrame(X_recidivism_test, index=X_recidivism_test_indices)
X_violence_train_df = pd.DataFrame(X_violence_train, index=X_violence_train_indices)
X_violence_test_df = pd.DataFrame(X_violence_test, index=X_violence_test_indices)

y_recidivism_severity_train_df = pd.DataFrame(y_recidivism_severity_train, index=y_recidivism_severity_train_indices)
y_recidivism_severity_test_df = pd.DataFrame(y_recidivism_severity_test, index=y_recidivism_severity_test_indices)
y_violence_severity_train_df = pd.DataFrame(y_violence_severity_train, index=y_violence_severity_train_indices)
y_violence_severity_test_df = pd.DataFrame(y_violence_severity_test, index=y_violence_severity_test_indices)

# Drop rows with NaN values in y and update X accordingly
y_recidivism_severity_train_df.dropna(inplace=True)
X_recidivism_train_df = X_recidivism_train_df.loc[y_recidivism_severity_train_df.index]

y_recidivism_severity_test_df.dropna(inplace=True)
X_recidivism_test_df = X_recidivism_test_df.loc[y_recidivism_severity_test_df.index]

y_violence_severity_train_df.dropna(inplace=True)
X_violence_train_df = X_violence_train_df.loc[y_violence_severity_train_df.index]

y_violence_severity_test_df.dropna(inplace=True)
X_violence_test_df = X_violence_test_df.loc[y_violence_severity_test_df.index]

# Drop rows with NaN values in X and update y accordingly
X_recidivism_train_df.dropna(inplace=True)
y_recidivism_severity_train_df = y_recidivism_severity_train_df.loc[X_recidivism_train_df.index]

X_recidivism_test_df.dropna(inplace=True)
y_recidivism_severity_test_df = y_recidivism_severity_test_df.loc[X_recidivism_test_df.index]

X_violence_train_df.dropna(inplace=True)
y_violence_severity_train_df = y_violence_severity_train_df.loc[X_violence_train_df.index]

X_violence_test_df.dropna(inplace=True)
y_violence_severity_test_df = y_violence_severity_test_df.loc[X_violence_test_df.index]

# Convert DataFrame back to NumPy arrays
X_recidivism_train = X_recidivism_train_df.values
X_recidivism_test = X_recidivism_test_df.values
X_violence_train = X_violence_train_df.values
X_violence_test = X_violence_test_df.values

X_recidivism_train_indices = X_recidivism_train_df.index
X_recidivism_test_indices = X_recidivism_test_df.index
X_violence_train_indices = X_violence_train_df.index
X_violence_test_indices = X_violence_test_df.index

y_recidivism_severity_train = y_recidivism_severity_train_df.values.ravel()
y_recidivism_severity_test = y_recidivism_severity_test_df.values.ravel()
y_violence_severity_train = y_violence_severity_train_df.values.ravel()
y_violence_severity_test = y_violence_severity_test_df.values.ravel()


# Label encode the 'race' column for adversarial debiasing
label_encoder = LabelEncoder()

# Recidivism 
race_train = recidivismData['race'].loc[X_recidivism_train_indices].values
race_recidivism_train_encoded = label_encoder.fit_transform(race_train)

# Violence
race_train = recidivismData['race'].loc[X_violence_train_indices].values
race_violence_train_encoded = label_encoder.fit_transform(race_train)

print("Recedivism misdeamonor level prediction without adversarial debiasing")
# Train recidivism pipeline without adversarial debiasing
pipeline = severity_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_severity_train, X_recidivism_test, 
             y_recidivism_severity_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=False, training_name='noAdversarialNoPostRecidivismSeverity')
pipeline.fit()
pipeline.predict()

# Evaluate bias for each racial group using false positives and false negatives
racialBiasDetectionMulti.evaluate_bias(X_recidivism_test, y_recidivism_severity_test, pipeline.get_final_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

print("Violence misdeamonor level prediction without adversarial debiasing")
# Violence classification pipeline without adversarial debiasing
pipelineviolence = severity_model_pipeline.CustomPipeline(X_violence_train, y_violence_severity_train, X_violence_test, 
             y_violence_severity_test, recidivismData, X_violence_test_indices, 'race', adversarial=False, training_name='noAdversarialNoPostViolenceSeverity')
pipelineviolence.fit()
pipelineviolence.predict()

# Evaluate bias for violence
racialBiasDetectionMulti.evaluate_bias(X_violence_test, y_violence_severity_test, pipeline.get_final_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

print("Recedivism misdeamonor level prediction with adversarial debiasing")
# Recidivism pipeline with adversarial debiasing
pipelineadversarial = severity_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_severity_train, X_recidivism_test, 
             y_recidivism_severity_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=True, training_name='adversarialNoPostRecidivismSeverity')
pipelineadversarial.fit(race_train=race_recidivism_train_encoded)
pipelineadversarial.predict()

# Evaluate bias for each racial group using false positives and false negatives (adversarial)
racialBiasDetectionMulti.evaluate_bias(X_recidivism_test, y_recidivism_severity_test, pipelineadversarial.get_final_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

print("Violence misdeamonor level prediction without adversarial debiasing")
# Violence classification with adversarial debiasing
pipelineadversarialviolence = severity_model_pipeline.CustomPipeline(X_violence_train, y_violence_severity_train, X_violence_test, 
             y_violence_severity_test, recidivismData, X_violence_test_indices, 'race', adversarial=True, training_name='adversarialNoPostViolenceSeverity')
pipelineadversarialviolence.fit(race_train=race_violence_train_encoded)
pipelineadversarialviolence.predict()

# Evaluate bias for violence (adversarial)
racialBiasDetectionMulti.evaluate_bias(X_violence_test, y_violence_severity_test, pipelineadversarialviolence.get_final_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# # <<----------------------- COMPAS Implementation for Comparison ---------------------->>

# Use only the decile_score and score_text for COMPAS processing
X_compas_recidivism = recidivismData[['decile_score', 'score_text']]

# Split data for recidivism classification
X_compas_violence = recidivismData[['v_decile_score', 'v_score_text']]
y_recidivism_severity = recidivismData['r_charge_degree']
y_violence_severity = recidivismData['vr_charge_degree']

# Define the static severity mapping without CO3 (since we are removing it)
severity_mapping = {'(F1)': 7, '(F2)': 6, '(F3)': 5, '(F6)': 4, '(F7)': 3, '(M1)': 2, '(M2)': 1, '(MO3)': 0}


# Filter out 'CO3' rows from both datasets
y_recidivism_severity = y_recidivism_severity[y_recidivism_severity != '(CO3)']
y_violence_severity = y_violence_severity[y_violence_severity != '(CO3)']

# Ensure X has matching indices after filtering out CO3 from y
X_compas_recidivism = X_compas_recidivism.loc[y_recidivism_severity.index]
X_compas_violence = X_compas_violence.loc[y_violence_severity.index]

# Apply the severity mapping to the filtered data
y_recidivism_severity = y_recidivism_severity.map(severity_mapping)
y_violence_severity = y_violence_severity.map(severity_mapping)

# Drop NaN values in case of any missing mappings
y_recidivism_severity = y_recidivism_severity.dropna().astype(int)
y_violence_severity = y_violence_severity.dropna().astype(int)

X_compas_recidivism = X_compas_recidivism.loc[y_recidivism_severity.index]
X_compas_violence = X_compas_violence.loc[y_violence_severity.index]


# Split data for recidivism classification
X_compas_recidivism_train, X_compas_recidivism_test, y_compas_recidivism_severity_train, y_compas_recidivism_severity_test = train_test_split(
    X_compas_recidivism, y_recidivism_severity, test_size=0.2, random_state=42)

# Split data for violence classification
X_compas_violence_train, X_compas_violence_test, y_compas_violence_severity_train, y_compas_violence_severity_test = train_test_split(
    X_compas_violence, y_violence_severity, test_size=0.2, random_state=42)

# Conserves the original dataset indexes before transforming it
X_compas_recidivism_test_indices = X_compas_recidivism_test.index
X_compas_recidivism_train_indices = X_compas_recidivism_train.index
X_compas_violence_test_indices = X_compas_violence_test.index
X_compas_violence_train_indices = X_compas_violence_train.index

y_compas_recidivism_severity_test_indices = y_compas_recidivism_severity_test.index
y_compas_recidivism_severity_train_indices = y_compas_recidivism_severity_train.index
y_compas_violence_severity_test_indices = y_compas_violence_severity_test.index
y_compas_violence_severity_train_indices = y_compas_violence_severity_train.index

# Convert to DataFrame to drop rows with NaN values
X_compas_recidivism_train_df = pd.DataFrame(X_compas_recidivism_train, index=X_compas_recidivism_train_indices)
X_compas_recidivism_test_df = pd.DataFrame(X_compas_recidivism_test, index=X_compas_recidivism_test_indices)
X_compas_violence_train_df = pd.DataFrame(X_compas_violence_train, index=X_compas_violence_train_indices)
X_compas_violence_test_df = pd.DataFrame(X_compas_violence_test, index=X_compas_violence_test_indices)

y_compas_recidivism_severity_train_df = pd.DataFrame(y_compas_recidivism_severity_train, index=y_recidivism_severity_train_indices)
y_compas_recidivism_severity_test_df = pd.DataFrame(y_compas_recidivism_severity_test, index=y_recidivism_severity_test_indices)
y_compas_violence_severity_train_df = pd.DataFrame(y_compas_violence_severity_train, index=y_violence_severity_train_indices)
y_compas_violence_severity_test_df = pd.DataFrame(y_compas_violence_severity_test, index=y_violence_severity_test_indices)

# Drop rows with NaN values in y and update X accordingly
y_compas_recidivism_severity_train_df.dropna(inplace=True)
X_compas_recidivism_train_df = X_compas_recidivism_train_df.loc[y_compas_recidivism_severity_train_df.index]

y_compas_recidivism_severity_test_df.dropna(inplace=True)
X_compas_recidivism_test_df = X_compas_recidivism_test_df.loc[y_compas_recidivism_severity_test_df.index]

y_compas_violence_severity_train_df.dropna(inplace=True)
X_compas_violence_train_df = X_compas_violence_train_df.loc[y_compas_violence_severity_train_df.index]

y_compas_violence_severity_test_df.dropna(inplace=True)
X_compas_violence_test_df = X_compas_violence_test_df.loc[y_compas_violence_severity_test_df.index]

# Drop rows with NaN values in X and update y accordingly
X_compas_recidivism_train_df.dropna(inplace=True)
y_compas_recidivism_severity_train_df = y_compas_recidivism_severity_train_df.loc[X_compas_recidivism_train_df.index]

X_compas_recidivism_test_df.dropna(inplace=True)
y_compas_recidivism_severity_test_df = y_compas_recidivism_severity_test_df.loc[X_compas_recidivism_test_df.index]

X_compas_violence_train_df.dropna(inplace=True)
y_compas_violence_severity_train_df = y_compas_violence_severity_train_df.loc[X_compas_violence_train_df.index]

X_compas_violence_test_df.dropna(inplace=True)
y_compas_violence_severity_test_df = y_compas_violence_severity_test_df.loc[X_compas_violence_test_df.index]

# Convert DataFrame back to NumPy arrays
X_compas_recidivism_train = X_compas_recidivism_train_df
X_compas_recidivism_test = X_compas_recidivism_test_df
X_compas_violence_train = X_compas_violence_train_df
X_compas_violence_test = X_compas_violence_test_df

y_compas_recidivism_severity_train = y_compas_recidivism_severity_train_df.values.ravel()
y_compas_recidivism_severity_test = y_compas_recidivism_severity_test_df.values.ravel()
y_compas_violence_severity_train = y_compas_violence_severity_train_df.values.ravel()
y_compas_violence_severity_test = y_compas_violence_severity_test_df.values.ravel()

X_compas_recidivism_train = X_compas_recidivism_train_df
X_compas_recidivism_test = X_compas_recidivism_test_df
X_compas_violence_train = X_compas_violence_train_df
X_compas_violence_test = X_compas_violence_test_df

X_compas_recidivism_test_indices = X_compas_recidivism_test.index
X_compas_recidivism_train_indices = X_compas_recidivism_train.index
X_compas_violence_test_indices = X_compas_violence_test.index
X_compas_violence_train_indices = X_compas_violence_train.index

# Preprocess the data for the original recidivism pipeline
X_compas_recidivism_train, X_compas_recidivism_test = recidivism_compass_preproccessing.preprocessor(X_compas_recidivism_train, X_compas_recidivism_test)
X_compas_violence_train, X_compas_violence_test = violence_compass_preproccessing.preprocessor(X_compas_violence_train, X_compas_violence_test)

# Recidivism classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_recidivism = severity_model_pipeline.CustomPipeline(X_compas_recidivism_train, y_compas_recidivism_severity_train, 
    X_compas_recidivism_test, y_compas_recidivism_severity_test, recidivismData, X_compas_recidivism_test_indices, 'race', adversarial=False, 
    training_name='noAdversarialNoPostTrainedCompasRecidivismSeverity')

print("Compas misdeamonor level prediction without adversarial debiasing")
pipeline_compas_recidivism.fit()
pipeline_compas_recidivism.predict()

# Evaluate bias for each racial group using COMPAS data for recidivism
racialBiasDetectionMulti.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_severity_test, 
    pipeline_compas_recidivism.get_final_pred(), recidivismData, X_compas_recidivism_test_indices, 'race')

print("Compas violence level prediction without adversarial debiasing")
# Violence classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_violence = severity_model_pipeline.CustomPipeline(X_compas_violence_train, y_compas_violence_severity_train, 
    X_compas_violence_test, y_compas_violence_severity_test, recidivismData, X_compas_violence_test_indices, 'race', adversarial=False,
    training_name='noAdversarialNoPostTrainedCompasViolenceSeverity')

pipeline_compas_violence.fit()
pipeline_compas_violence.predict()

# Evaluate bias for each racial group using COMPAS data for violence
racialBiasDetectionMulti.evaluate_bias(X_compas_violence_test, y_compas_violence_severity_test, 
    pipeline_compas_violence.get_final_pred(), recidivismData, X_compas_violence_test_indices, 'race')

# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>
# <<---------------------------------------------------------------------------------------------------->>

# This is for classification of the time it takes for recidivism to occur again
# This is for classification of the time it takes for recidivism to occur again
# This is for classification of the time it takes for recidivism to occur again
# This is for classification of the time it takes for recidivism to occur again

import models.recidivism.date.model_pipeline as date_model_pipeline

# 1. Convert date columns to datetime, ensuring proper format
recidivism_data_copy = recidivismData.copy()  # We work on a copy to avoid modifying the original data

X_recidivism = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'score_text', 'decile_score',
                               'c_charge_degree']]
X_violence = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'v_score_text', 'v_decile_score',
                               'c_charge_degree']]
# Convert the 'c_jail_out', 'r_offense_date', and 'vr_offense_date' to datetime format
recidivism_data_copy['c_jail_out'] = pd.to_datetime(recidivism_data_copy['c_jail_out'], dayfirst=True, errors='coerce').dt.date
recidivism_data_copy['r_offense_date'] = pd.to_datetime(recidivism_data_copy['r_offense_date'], dayfirst=True, errors='coerce').dt.date
recidivism_data_copy['vr_offense_date'] = pd.to_datetime(recidivism_data_copy['vr_offense_date'], dayfirst=True, errors='coerce').dt.date

# 2. Compute the difference in days for recidivism and violence, separately

# For recidivism (c_jail_out and r_offense_date)
recidivism_clean = recidivism_data_copy.dropna(subset=['c_jail_out', 'r_offense_date']).copy()
recidivism_clean['days_until_recidivism'] = (recidivism_clean['r_offense_date'] - recidivism_clean['c_jail_out']).apply(lambda x: x.days)

# For violence (c_jail_out and vr_offense_date)
violence_clean = recidivism_data_copy.dropna(subset=['c_jail_out', 'vr_offense_date']).copy()
violence_clean['days_until_violence'] = (violence_clean['vr_offense_date'] - violence_clean['c_jail_out']).apply(lambda x: x.days)

# 3. Handle NaN values and ensure index consistency between X and Y
# Now drop any rows with NaN in the 'days_until_recidivism' and 'days_until_violence'
recidivism_clean.dropna(subset=['days_until_recidivism'], inplace=True)
violence_clean.dropna(subset=['days_until_violence'], inplace=True)

# Define X and Y (target) for recidivism
X_recidivism = X_recidivism.loc[recidivism_clean.index]  # Align X with recidivism index
y_recidivism_date = recidivism_clean['days_until_recidivism']

# Define X and Y (target) for violence
X_violence = X_violence.loc[violence_clean.index]  # Align X with violence index
y_violence_date = violence_clean['days_until_violence']

# 3. Handle NaN values and ensure index consistency between X and Y
# Now drop any rows with NaN in the 'days_until_recidivism' and 'days_until_violence'
recidivism_clean.dropna(subset=['days_until_recidivism'], inplace=True)
violence_clean.dropna(subset=['days_until_violence'], inplace=True)

# Ensure X and y have consistent indices
X_recidivism = X_recidivism.loc[recidivism_clean.index]
y_recidivism_date = recidivism_clean['days_until_recidivism']

X_violence = X_violence.loc[violence_clean.index]
y_violence_date = violence_clean['days_until_violence']

# Split data for recidivism and violence classification
X_recidivism_train, X_recidivism_test, y_recidivism_date_train, y_recidivism_date_test = train_test_split(
    X_recidivism, y_recidivism_date, test_size=0.2, random_state=42)
X_violence_train, X_violence_test, y_violence_date_train, y_violence_date_test = train_test_split(
    X_violence, y_violence_date, test_size=0.2, random_state=42)

# Label encode 'race' column for adversarial debiasing
label_encoder = LabelEncoder()
race_recidivism_train_encoded = label_encoder.fit_transform(X_recidivism_train['race'].loc[X_recidivism_train.index].values)
race_violence_train_encoded = label_encoder.fit_transform(X_violence_train['race'].loc[X_violence_train.index].values)

X_recidivism_test_indices = X_recidivism_test.index
X_recidivism_train_indices = X_recidivism_train.index
X_violence_test_indices = X_violence_test.index
X_violence_train_indices = X_violence_train.index


X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)
X_violence_train, X_violence_test = violence_preproccessing.preprocessor(X_violence_train, X_violence_test)

X_recidivism_train_df = pd.DataFrame(X_recidivism_train, index=X_recidivism_train_indices)
X_recidivism_test_df = pd.DataFrame(X_recidivism_test, index=X_recidivism_test_indices)
X_violence_train_df = pd.DataFrame(X_violence_train, index=X_violence_train_indices)
X_violence_test_df = pd.DataFrame(X_violence_test, index=X_violence_test_indices)

X_recidivism_test_indices = X_recidivism_test_df.index
X_recidivism_train_indices = X_recidivism_train_df.index
X_violence_test_indices = X_violence_test_df.index
X_violence_train_indices = X_violence_train_df.index

print("Recedivism day prediction without adversarial debiasing")
# Train Recidivism Pipeline (without adversarial debiasing)
pipeline = date_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_date_train, X_recidivism_test, 
             y_recidivism_date_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=False
             , training_name='noAdversarialNoPostRecidivismDate')
pipeline.fit()
pipeline.predict()

print("Violence day prediction without adversarial debiasing")
# Train Violence Pipeline (without adversarial debiasing)
pipelineviolence = date_model_pipeline.CustomPipeline(X_violence_train, y_violence_date_train, X_violence_test, 
             y_violence_date_test, recidivismData, X_violence_test_indices, 'race', adversarial=False,
              training_name='noAdversarialNoPostViolenceDate')
pipelineviolence.fit()
pipelineviolence.predict()

print("Recedivism day prediction with adversarial debiasing")
# Train Recidivism Pipeline (with adversarial debiasing)
pipelineadversarial = date_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_date_train, X_recidivism_test, 
             y_recidivism_date_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=True, training_name='adversarialNoPostRecidivismDate')
pipelineadversarial.fit(race_train=race_recidivism_train_encoded)
pipelineadversarial.predict()

print("Violence day prediction with adversarial debiasing")
# Train Violence Pipeline (with adversarial debiasing)
pipelineadversarialviolence = date_model_pipeline.CustomPipeline(X_violence_train, y_violence_date_train, X_violence_test, 
             y_violence_date_test, recidivismData, X_violence_test_indices, 'race', adversarial=True, training_name='adversarialNoPostViolenceDate')
pipelineadversarialviolence.fit(race_train=race_violence_train_encoded)
pipelineadversarialviolence.predict()

# # <<----------------------- COMPAS Implementation for Comparison ---------------------->>

# Use only the decile_score and score_text for COMPAS processing
X_compas_recidivism = recidivismData[['decile_score', 'score_text']]

# Split data for recidivism classification
X_compas_violence = recidivismData[['v_decile_score', 'v_score_text']]


# 1. Convert date columns to datetime, ensuring proper format
recidivism_data_copy = recidivismData.copy()  # We work on a copy to avoid modifying the original data

# Convert the 'c_jail_out', 'r_offense_date', and 'vr_offense_date' to datetime format
recidivism_data_copy['c_jail_out'] = pd.to_datetime(recidivism_data_copy['c_jail_out'], dayfirst=True, errors='coerce').dt.date
recidivism_data_copy['r_offense_date'] = pd.to_datetime(recidivism_data_copy['r_offense_date'], dayfirst=True, errors='coerce').dt.date
recidivism_data_copy['vr_offense_date'] = pd.to_datetime(recidivism_data_copy['vr_offense_date'], dayfirst=True, errors='coerce').dt.date

# 2. Compute the difference in days for recidivism and violence, separately

# For recidivism (c_jail_out and r_offense_date)
recidivism_clean = recidivism_data_copy.dropna(subset=['c_jail_out', 'r_offense_date']).copy()
recidivism_clean['days_until_recidivism'] = (recidivism_clean['r_offense_date'] - recidivism_clean['c_jail_out']).apply(lambda x: x.days)

# For violence (c_jail_out and vr_offense_date)
violence_clean = recidivism_data_copy.dropna(subset=['c_jail_out', 'vr_offense_date']).copy()
violence_clean['days_until_violence'] = (violence_clean['vr_offense_date'] - violence_clean['c_jail_out']).apply(lambda x: x.days)

# 3. Handle NaN values and ensure index consistency between X and Y
# Now drop any rows with NaN in the 'days_until_recidivism' and 'days_until_violence'
recidivism_clean.dropna(subset=['days_until_recidivism'], inplace=True)
violence_clean.dropna(subset=['days_until_violence'], inplace=True)

# Define X and Y (target) for recidivism
X_compas_recidivism = X_compas_recidivism.loc[recidivism_clean.index]  # Align X with recidivism index
y_compas_recidivism_date = recidivism_clean['days_until_recidivism']

# Define X and Y (target) for violence
X_compas_violence = X_compas_violence.loc[violence_clean.index]  # Align X with violence index
y_compas_violence_date = violence_clean['days_until_violence']

# 3. Handle NaN values and ensure index consistency between X and Y
# Now drop any rows with NaN in the 'days_until_recidivism' and 'days_until_violence'
recidivism_clean.dropna(subset=['days_until_recidivism'], inplace=True)
violence_clean.dropna(subset=['days_until_violence'], inplace=True)

# Ensure X and y have consistent indices
X_compas_recidivism = X_compas_recidivism.loc[recidivism_clean.index]
y_compas_recidivism_date = recidivism_clean['days_until_recidivism']

X_compas_violence = X_compas_violence.loc[violence_clean.index]
y_compas_violence_date = violence_clean['days_until_violence']

# Split data for recidivism and violence classification
X_compas_recidivism_train, X_compas_recidivism_test, y_compas_recidivism_date_train, y_compas_recidivism_date_test = train_test_split(
    X_compas_recidivism, y_compas_recidivism_date, test_size=0.2, random_state=42)
X_compas_violence_train, X_compas_violence_test, y_compas_violence_date_train, y_compas_violence_date_test = train_test_split(
    X_compas_violence, y_compas_violence_date, test_size=0.2, random_state=42)

X_compas_recidivism_test_indices = X_compas_recidivism_test.index
X_compas_recidivism_train_indices = X_compas_recidivism_train.index
X_compas_violence_test_indices = X_compas_violence_test.index
X_compas_violence_train_indices = X_compas_violence_train.index


X_compas_recidivism_train, X_compas_recidivism_test = recidivism_compass_preproccessing.preprocessor(X_compas_recidivism_train, X_compas_recidivism_test)
X_compas_violence_train, X_compas_violence_test = violence_compass_preproccessing.preprocessor(X_compas_violence_train, X_compas_violence_test)

X_compas_recidivism_train_df = pd.DataFrame(X_compas_recidivism_train, index=X_compas_recidivism_train_indices)
X_compas_recidivism_test_df = pd.DataFrame(X_compas_recidivism_test, index=X_compas_recidivism_test_indices)
X_compas_violence_train_df = pd.DataFrame(X_compas_violence_train, index=X_compas_violence_train_indices)
X_compas_violence_test_df = pd.DataFrame(X_compas_violence_test, index=X_compas_violence_test_indices)

X_compas_recidivism_test_indices = X_compas_recidivism_test_df.index
X_compas_recidivism_train_indices = X_compas_recidivism_train_df.index
X_compas_violence_test_indices = X_compas_violence_test_df.index
X_compas_violence_train_indices = X_compas_violence_train_df.index

print("Compas Recedivism day prediction without adversarial debiasing")
# Train Recidivism Pipeline (without adversarial debiasing)
pipeline = date_model_pipeline.CustomPipeline(X_compas_recidivism_train, y_compas_recidivism_date_train, X_compas_recidivism_test, 
             y_compas_recidivism_date_test, recidivismData, X_compas_recidivism_test_indices, 'race', adversarial=False
             , training_name='noAdversarialNoPostTrainedCompasRecidivismDate')
pipeline.fit()
pipeline.predict()

print("Compas Violence day prediction without adversarial debiasing")
# Train Violence Pipeline (without adversarial debiasing)
pipelineviolence = date_model_pipeline.CustomPipeline(X_compas_violence_train, y_compas_violence_date_train, X_compas_violence_test, 
             y_compas_violence_date_test, recidivismData, X_compas_violence_test_indices, 'race', adversarial=False, 
             training_name='noAdversarialNoPostTrainedCompasViolenceDate')
pipelineviolence.fit()
pipelineviolence.predict()

# <<---------------------------------------------------------------------------------------------------->>