import pandas as pd
from sklearn.model_selection import train_test_split
import data.income.data as data
import data.income.preproccessing as preproccessing
import models.income.model_pipeline as model_pipeline
import evaluations.racialBiasDetection as racialBiasDetection
import postprocessing.classification.equalized_odds as equalized_odds
import evaluations.racialBiasDetection as racialBiasDetection

# All the parameters that are considered as independent variables/factors
# Essentially all the factors in the dataset except the target variable (income)
data = data.get_data()
X = data[['age', 'workclass','education', 'education.num', 'marital.status', 
          'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
          'hours.per.week', 'native.country']]
# The target variable. Which is the income column
y = data['income']

# Convert the classification of >50K and <=50K to 1 and 0 respectively
# ML models require numerical boolean values typically to work for classification algorithms
y = y.map({'>50K': 1, '<=50K': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conserves the original dataset indexes before transforming it to a csr_matrix by preprocessor
X_test_indices = X_test.index
X_train_indices = X_train.index

X_train, X_test = preproccessing.preprocessor(X_train, X_test)

# Then train the pipeline. Produce the accuracies that are geneeral and for racial groups
pipeline = model_pipeline.CustomPipeline()
pipeline.fit(X_train, y_train, X_test, y_test, data, X_test_indices)

# Evaluate bias for each racial group using false positive and false negatives
racialBiasDetection.evaluate_bias(X_test, y_test, model_pipeline.get_final_pred(), data, X_test_indices, 'race')

# Evaluate the model with post-processing to ensure fairness
equalized_odds.equalize(data, pipeline, X_train, y_train, X_test, y_test, X_test_indices, X_train_indices, 'race')

# Evaluate raical bias with this post-proccessing
racialBiasDetection.evaluate_bias(X_test, y_test, 
        equalized_odds.get_y_pred_fixed(), data, X_test_indices, 'race')