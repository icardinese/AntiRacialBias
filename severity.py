y_recidivism_severity = recidivismData['r_charge_degree']
y_violence_severity = recidivismData['vr_charge_degree']

# Split data for recidivism and violence classification
X_recidivism_train, X_recidivism_test, y_recidivism_severity_train, y_recidivism_severity_test = train_test_split(
    X_recidivism, y_recidivism_severity, test_size=0.2, random_state=42)
X_violence_train, X_violence_test, y_violence_severity_train, y_violence_severity_test = train_test_split(
    X_violence, y_violence_severity, test_size=0.2, random_state=42)

# Conserves the original dataset indexes before transforming it
X_recidivism_test_indices = X_recidivism_test.index
X_recidivism_train_indices = X_recidivism_train.index
X_violence_test_indices = X_violence_test.index
X_violence_train_indices = X_violence_train.index

y_recidivism_severity_test_indices = y_recidivism_severity_test.index
y_recidivism_severity_train_indices = y_recidivism_severity_train.index
y_violence_severity_test_indices = y_violence_severity_test.index
y_violence_severity_train_indices = y_violence_severity_train.index

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

y_recidivism_severity_train = y_recidivism_severity_train_df.values.ravel()
y_recidivism_severity_test = y_recidivism_severity_test_df.values.ravel()
y_violence_severity_train = y_violence_severity_train_df.values.ravel()
y_violence_severity_test = y_violence_severity_test_df.values.ravel()

# Preprocess the data for the original recidivism pipeline
X_recidivism_train, X_recidivism_test = recidivism_preproccessing.preprocessor(X_recidivism_train, X_recidivism_test)
X_violence_train, X_violence_test = violence_preproccessing.preprocessor(X_violence_train, X_violence_test)

# Label encode the 'race' column for adversarial debiasing
label_encoder = LabelEncoder()

# Recidivism 
race_train = recidivismData['race'].loc[X_recidivism_train_indices].values
race_recidivism_train_encoded = label_encoder.fit_transform(race_train)

# Violence
race_train = recidivismData['race'].loc[X_violence_train_indices].values
race_violence_train_encoded = label_encoder.fit_transform(race_train)

# Train recidivism pipeline without adversarial debiasing
pipeline = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_severity_train, X_recidivism_test, 
             y_recidivism_severity_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=False)
pipeline.fit()
pipeline.predict()

# Evaluate bias for each racial group using false positives and false negatives
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_severity_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

# Violence classification pipeline without adversarial debiasing
pipelineviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_severity_train, X_violence_test, 
             y_violence_severity_test, recidivismData, X_violence_test_indices, 'race', adversarial=False)
pipelineviolence.fit()
pipelineviolence.predict()

# Evaluate bias for violence
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_severity_test, pipeline.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# Recidivism pipeline with adversarial debiasing
pipelineadversarial = classification_model_pipeline.CustomPipeline(X_recidivism_train, y_recidivism_severity_train, X_recidivism_test, 
             y_recidivism_severity_test, recidivismData, X_recidivism_test_indices, 'race', adversarial=True)
pipelineadversarial.fit(race_train=race_recidivism_train_encoded)
pipelineadversarial.predict()

# Evaluate bias for each racial group using false positives and false negatives (adversarial)
racialBiasDetection.evaluate_bias(X_recidivism_test, y_recidivism_severity_test, pipelineadversarial.get_final_binary_pred(),
                                   recidivismData, X_recidivism_test_indices, 'race')

# Violence classification with adversarial debiasing
pipelineadversarialviolence = classification_model_pipeline.CustomPipeline(X_violence_train, y_violence_severity_train, X_violence_test, 
             y_violence_severity_test, recidivismData, X_violence_test_indices, 'race', adversarial=True)
pipelineadversarialviolence.fit(race_train=race_violence_train_encoded)
pipelineadversarialviolence.predict()

# Evaluate bias for violence (adversarial)
racialBiasDetection.evaluate_bias(X_violence_test, y_violence_severity_test, pipelineadversarialviolence.get_final_binary_pred(),
                                   recidivismData, X_violence_test_indices, 'race')

# # <<----------------------- COMPAS Implementation for Comparison ---------------------->>

# Use only the decile_score and score_text for COMPAS processing
X_compas_recidivism = recidivismData[['decile_score', 'score_text']]

# Split data for recidivism classification
X_compas_violence = recidivismData[['v_decile_score', 'v_score_text']]
y_recidivism_severity = recidivismData['r_charge_degree']
y_violence_severity = recidivismData['vr_charge_degree']

# Split data for recidivism classification
X_compas_recidivism_train, X_compas_recidivism_test, y_compas_recidivism_severity_train, y_compas_recidivism_severity_test = train_test_split(
    X_compas_recidivism, y_recidivism_severity, test_size=0.2, random_state=42)

# Split data for violence classification
X_compas_violence_train, X_compas_violence_test, y_compas_violence_severity_train, y_compas_violence_severity_test = train_test_split(
    X_compas_violence, y_violence_severity, test_size=0.2, random_state=42)

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
y_compas_recidivism_severity_train = y_compas_recidivism_severity_train.loc[X_compas_recidivism_train_indices]
y_compas_recidivism_severity_test = y_compas_recidivism_severity_test.loc[X_compas_recidivism_test_indices]
y_compas_violence_severity_train = y_compas_violence_severity_train.loc[X_compas_violence_train_indices]
y_compas_violence_severity_test = y_compas_violence_severity_test.loc[X_compas_violence_test_indices]


# Recidivism classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_recidivism = classification_model_pipeline.CustomPipeline(X_compas_recidivism_train, y_compas_recidivism_severity_train, 
    X_compas_recidivism_test, y_compas_recidivism_severity_test, recidivismData, X_compas_recidivism_test_indices, 'race', adversarial=False)

pipeline_compas_recidivism.fit()
pipeline_compas_recidivism.predict()

# Evaluate bias for each racial group using COMPAS data for recidivism
racialBiasDetection.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_severity_test, 
    pipeline_compas_recidivism.get_final_binary_pred(), recidivismData, X_compas_recidivism_test_indices, 'race')

# Violence classification pipeline using COMPAS data without adversarial debiasing
pipeline_compas_violence = classification_model_pipeline.CustomPipeline(X_compas_violence_train, y_compas_violence_severity_train, 
    X_compas_violence_test, y_compas_violence_severity_test, recidivismData, X_compas_violence_test_indices, 'race', adversarial=False)

pipeline_compas_violence.fit()
pipeline_compas_violence.predict()

# Evaluate bias for each racial group using COMPAS data for violence
racialBiasDetection.evaluate_bias(X_compas_violence_test, y_compas_violence_severity_test, 
    pipeline_compas_violence.get_final_binary_pred(), recidivismData, X_compas_violence_test_indices, 'race')

# Recidivism classification based on pure decile score
X_compas_recidivism_test_original = recidivismData.loc[X_compas_recidivism_test_indices]
recidivism_racial_groups = X_compas_recidivism_test_original['race'].unique()

final_compass_recidivism_binary = (X_compas_recidivism_test_original['decile_score'] > 5).astype(int)
y_compass_recidivism_test_series = pd.Series(y_compas_recidivism_severity_test, index=X_compas_recidivism_test_indices)
final_compass_recidivism_pred_series = pd.Series(final_compass_recidivism_binary, index=X_compas_recidivism_test_indices)

overall_accuracy = accuracy_score(y_compas_recidivism_severity_test, final_compass_recidivism_binary)
print(f"Overall accuracy: {overall_accuracy * 100.0}%")

for group in recidivism_racial_groups:
        group_indices = X_compas_recidivism_test_original[X_compas_recidivism_test_original['race'] == group].index
        group_y_test = y_compass_recidivism_test_series.loc[group_indices]
        group_y_pred = final_compass_recidivism_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

racialBiasDetection.evaluate_bias(X_compas_recidivism_test, y_compas_recidivism_severity_test, 
    final_compass_recidivism_binary, recidivismData, X_compas_recidivism_test_indices, 'race')

X_compas_violence_test_original = recidivismData.loc[X_compas_violence_test_indices]
violence_racial_groups = X_compas_violence_test_original['race'].unique()

final_compass_violence_binary = (recidivismData.loc[X_compas_violence_test_indices]['v_decile_score'] > 5).astype(int)
y_compass_violence_test_series = pd.Series(y_compas_violence_severity_test, index=X_compas_violence_test_indices)
final_compass_violence_pred_series = pd.Series(final_compass_violence_binary, index=X_compas_violence_test_indices)

overall_accuracy = accuracy_score(y_compas_violence_severity_test, final_compass_violence_binary)
print(f"Overall accuracy: {overall_accuracy * 100.0}%")

for group in violence_racial_groups:
        group_indices = X_compas_violence_test_original[X_compas_violence_test_original['race'] == group].index
        group_y_test = y_compass_violence_test_series.loc[group_indices]
        group_y_pred = final_compass_violence_pred_series.loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

racialBiasDetection.evaluate_bias(X_compas_violence_test, y_compas_violence_severity_test, 
    final_compass_violence_binary, recidivismData, X_compas_violence_test_indices, 'race')

# <<---------------------------------------------------------------------------------------------------->>