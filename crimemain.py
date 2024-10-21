import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import data.recidivism.data as recidivism_data
import data.recidivism.compasspreproccessing as compass_preprocessing  # Import your COMPAS preprocessor
import data.recidivism.preproccessing as recidivism_preproccessing
import models.recidivism.classification.model_pipeline as classification_model_pipeline
import evaluations.racialBiasDetection as racialBiasDetection
import postprocessing.postprocessing as postprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# <<---------------------------------------------->>
# This is for recidivism classification!

# Load recidivism data
recidivismData = recidivism_data.get_data()

# Define recidivism X and y
X_recidivism = recidivismData[['age', 'juv_fel_count', 'juv_misd_count', 
                               'juv_other_count', 'priors_count', 
                               'days_b_screening_arrest', 'c_days_from_compas', 
                               'sex', 'race', 'score_text', 'decile_score']]
y_recidivism_classification = recidivismData['is_recid']
y_violence_classification = recidivismData['is_violent_recid']

# Split data for recidivism and violence classification
X_recidivism_train, X_recidivism_test, y_recidivism_classification_train, y_recidivism_classification_test = train_test_split(
    X_recidivism, y_recidivism_classification, test_size=0.2, random_state=42)
X_violence_train, X_violence_test, y_violence_classification_train, y_violence_classification_test = train_test_split(
    X_recidivism, y_violence_classification, test_size=0.2, random_state=42)

# Conserves the original dataset indexes before transforming it
X_recidivism_test_indices = X_recidivism_test.index
X_recidivism_train_indices = X_recidivism_train.index
X_violence_test_indices = X_violence_test.index
X_violence_train_indices = X_violence_train.index

# Preprocess the data for the original recidivism pipeline
X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)
X_violence_train, X_violence_test = recidivism_preproccessing.preprocessor(X_violence_train, X_violence_test)

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
y_recidivism_classification_train = y_recidivism_classification_train.loc[X_recidivism_train_df.index]
y_recidivism_classification_test = y_recidivism_classification_test.loc[X_recidivism_test_df.index]
y_violence_classification_train = y_violence_classification_train.loc[X_violence_train_df.index]
y_violence_classification_test = y_violence_classification_test.loc[X_violence_test_df.index]



# Label encode the 'race' column for adversarial debiasing
label_encoder = LabelEncoder()
race_train = recidivismData['race'].loc[X_recidivism_train_df.index].values
race_train_encoded = label_encoder.fit_transform(race_train)

# Train recidivism pipeline without adversarial debiasing
pipeline = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_classification_train, X_recidivism_test, 
             y_recidivism_classification_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=False)
pipeline.fit()
pipeline.predict()

# Evaluate bias for each racial group using false positives and false negatives
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_classification_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

# Violence classification pipeline without adversarial debiasing
pipelineviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_classification_train, X_violence_test, 
             y_violence_classification_test, recidivismData, X_violence_test_indices, 'race', adversarial=False)
pipelineviolence.fit()
pipelineviolence.predict()

# Evaluate bias for violence
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_classification_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# Recidivism pipeline with adversarial debiasing
pipelineadversarial = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_classification_train, X_recidivism_test, 
             y_recidivism_classification_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=True)
pipelineadversarial.fit(race_train=race_train_encoded)
pipelineadversarial.predict()

# Evaluate bias for each racial group using false positives and false negatives (adversarial)
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_classification_test, pipelineadversarial.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

# Violence classification with adversarial debiasing
pipelineadversarialviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_classification_train, X_violence_test, 
             y_violence_classification_test, recidivismData, X_violence_test_indices, 'race', adversarial=True)
pipelineadversarialviolence.fit(race_train=race_train_encoded)
pipelineadversarialviolence.predict()

# Evaluate bias for violence (adversarial)
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_classification_test, pipelineadversarialviolence.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# <<----------------------- COMPAS Implementation for Comparison ---------------------->>

# Use only the decile_score and score_text for COMPAS processing
X_compas = recidivismData[['decile_score', 'score_text']]

# Split data for recidivism classification
X_compas = recidivismData[['decile_score', 'score_text']]
y_recidivism_classification = recidivismData['is_recid']
y_violence_classification = recidivismData['is_violent_recid']

# Split data for recidivism classification
X_compas_recidivism_train, X_compas_recidivism_test, y_compas_recidivism_classification_train, y_compas_recidivism_classification_test = train_test_split(
    X_compas, y_recidivism_classification, test_size=0.2, random_state=42)

# Split data for violence classification
X_compas_violence_train, X_compas_violence_test, y_compas_violence_classification_train, y_compas_violence_classification_test = train_test_split(
    X_compas, y_violence_classification, test_size=0.2, random_state=42)

X_compas_violence_test_indices = X_compas_violence_test.index
X_compas_violence_train_indices = X_compas_violence_train.index
X_compas_recidivism_test_indices = X_compas_recidivism_test.index
X_compas_recidivism_train_indices = X_compas_recidivism_train.index

# Preprocess the COMPAS data
X_compas_recidivism_train, X_compas_recidivism_test = compass_preprocessing.preprocessor(X_compas_recidivism_train, X_compas_recidivism_test)
X_compas_violence_train, X_compas_violence_test = compass_preprocessing.preprocessor(X_compas_violence_train, X_compas_violence_test)

# Convert to DataFrame to drop rows with NaN values
X_compas_recidivism_train_df = pd.DataFrame(X_recidivism_train, index=X_recidivism_train_indices)
X_compas_recidivism_test_df = pd.DataFrame(X_recidivism_test, index=X_recidivism_test_indices)
X_compas_violence_train_df = pd.DataFrame(X_violence_train, index=X_violence_train_indices)
X_compas_violence_test_df = pd.DataFrame(X_violence_test, index=X_violence_test_indices)

# Drop rows with NaN values
X_compas_recidivism_train_df = X_compas_recidivism_train_df.dropna(inplace=True)
X_compas_recidivism_test_df = X_compas_recidivism_test_df.dropna(inplace=True)
X_compas_violence_train_df = X_compas_violence_train_df.dropna(inplace=True)
X_compas_violence_test_df = X_compas_violence_test_df.dropna(inplace=True)

# Update indices
X_compas_recidivism_train_indices = X_recidivism_train_df.index
X_compas_recidivism_test_indices = X_recidivism_test_df.index
X_compas_violence_train_indices = X_violence_train_df.index
X_compas_violence_test_indices = X_violence_test_df.index

X_compas_recidivism_train = X_recidivism_train_df.values
X_compas_recidivism_test = X_recidivism_test_df.values
X_compas_violence_train = X_violence_train_df.values
X_compas_violence_test = X_violence_test_df.values

# Drop corresponding rows in y data
y_compas_recidivism_classification_train = y_recidivism_classification_train.loc[X_recidivism_train_indices]
y_compas_recidivism_classification_test = y_recidivism_classification_test.loc[X_recidivism_test_indices]
y_compas_violence_classification_train = y_violence_classification_train.loc[X_violence_train_indices]
y_compas_violence_classification_test = y_violence_classification_test.loc[X_violence_test_indices]


# Recidivism classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_recidivism = classification_model_pipeline.CustomPipeline(X_compas_recidivism_train, y_compas_recidivism_classification_train, 
    X_compas_recidivism_test, y_compas_recidivism_classification_test, recidivismData, X_compas_recidivism_test_indices, 'race', adversarial=False)

pipeline_compas_recidivism.fit()
pipeline_compas_recidivism.predict()

# Evaluate bias for each racial group using COMPAS data for recidivism
racialBiasDetection.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_classification_test, 
    pipeline_compas_recidivism.get_final_binary_pred(), recidivismData, X_compas_recidivism_test_indices, 'race')

# Violence classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_violence = classification_model_pipeline.CustomPipeline(X_compas_violence_train, y_compas_violence_classification_train, 
    X_compas_violence_test, y_compas_violence_classification_test, recidivismData, X_compas_violence_test_indices, 'race', adversarial=False)

pipeline_compas_violence.fit()
pipeline_compas_violence.predict()

# Evaluate bias for each racial group using COMPAS data for violence
racialBiasDetection.evaluate_bias(X_compas_violence_test, y_compas_violence_classification_test, 
    pipeline_compas_violence.get_final_binary_pred(), recidivismData, X_compas_violence_test_indices, 'race')

# Recidivism classification based on pure decile score
racial_groups = X_compas_recidivism_test['race'].unique()

final_compass_recidivism_binary = (X_compas_recidivism_test['decile_score'] > 5).astype(int)
y_compass_recidivism_test_series = pd.Series(y_compas_recidivism_classification_test, index=X_compas_recidivism_test_indices)
final_compass_recidivism_pred_series = pd.Series(final_compass_recidivism_binary, index=X_compas_recidivism_test_indices)

for group in racial_groups:
        group_indices = X_compas_recidivism_test[X_compas_recidivism_test['race'] == group].index
        group_y_test = y_compass_recidivism_test_series.loc[group_indices]
        group_y_pred = final_compass_recidivism_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

final_compass_violence_binary = (X_compas_violence_test['decile_score'] > 5).astype(int)
y_compass_violence_test_series = pd.Series(y_compas_violence_classification_test, index=X_compas_violence_test_indices)
final_compass_violence_pred_series = pd.Series(final_compass_violence_binary, index=X_compas_violence_test_indices)

for group in racial_groups:
        group_indices = X_compas_violence_test[X_compas_violence_test['race'] == group].index
        group_y_test = y_compass_violence_test_series.loc[group_indices]
        group_y_pred = final_compass_violence_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")
