from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Define the column transformer with imputation
# One-hot encoding of categorial variables. 
# For more information look here: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

def preprocessor(X_train, X_test):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']),
            ('cat_ord', OrdinalEncoder(), ['education']),
            ('cat_nom', OneHotEncoder(handle_unknown='ignore'), ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country'])
        ]
    )

    # Conserves the original dataset indexes before transforming it to a csr_matrix by preprocessor
    # X_test_indices = X_test.index
    # X_train_indices = X_train.index

    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert spacrse matrices to dense arrays so model can interpret the data
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test