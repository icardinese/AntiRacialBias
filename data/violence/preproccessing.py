from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Define the column transformer with imputation
# One-hot encoding of categorial variables. 
# For more information look here: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

def preprocessor(X_train, X_test):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                                                     'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 
                                                     'v_decile_score']),
            ('cat_ord', OrdinalEncoder(), ['v_score_text', 'c_charge_degree']),
            ('cat_nom', OneHotEncoder(handle_unknown='ignore'), ['sex', 'race'])
        ]
    )

    # Conserves the original dataset indexes before transforming it to a csr_matrix by preprocessor
    # X_test_indices = X_test.index
    # X_train_indices = X_train.index

    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.fit_transform(X_test)

    # Convert spacrse matrices to dense arrays so model can interpret the data
    # X_train = X_train.toarray()
    # X_test = X_test.toarray()
    return X_train, X_test