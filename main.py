import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import data
import model
import racialBiasDetection
import postProccessing

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

# Define the column transformer with imputation
# One-hot encoding of categorial variables. 
# For more information look here: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']),
        ('cat_ord', OrdinalEncoder(), ['education']),
        ('cat_nom', OneHotEncoder(handle_unknown='ignore'), ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country'])
    ]
)

# Conserves the original dataset indexes before transforming it to a csr_matrix by preprocessor
X_test_indices = X_test.index
X_train_indices = X_train.index

# Apply transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Convert spacrse matrices to dense arrays so model can interpret the data
X_train = X_train.toarray()
X_test = X_test.toarray()

# Then train the model. Produce the accuracies that are geneeral and for racial groups
model.XGBoost(X_train, y_train, X_test, y_test, data, X_test_indices)

# Evaluate bias for each racial group using false positive and false negatives
racialBiasDetection.evaluate_bias(X_test, y_test, model.get_y_pred(), data, X_test_indices)

# Evaluate the model with post-processing to ensure fairness
postProccessing.equalize(data, model.get_best_model(), X_train, y_train, X_test, y_test, X_test_indices, X_train_indices)

# Evaluate raical bias with this post-proccessing
racialBiasDetection.evaluate_bias(X_test, y_test, 
        postProccessing.get_y_pred_fixed(), data, X_test_indices)