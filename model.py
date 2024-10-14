from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

model = XGBClassifier()

best_model = None
y_pred = None

def XGBoost(x_train, y_train, x_test, y_test, data, x_test_indices):

    # This is for hyperparameter tuning the model. <<--------------------------------->>
    # Lists possible parameters to test with different choice of values.
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.3, 0.7, 1.0]
    }

    # Comparitevely through accuracy score the best parameters are selected.
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='accuracy', n_iter=20, random_state=42)
    # Training the model with different parameters.
    random_search.fit(x_train, y_train)
    best_params = random_search.best_params_
    # The best parameters are printed out based on choices in the parameters of param_grid.
    print(f"Best parameters: {best_params}")
    # <<--------------------------------->>

    # The ideal model based on hypertuning is selected.
    global best_model
    best_model = XGBClassifier(**best_params)
    best_model.fit(x_train, y_train)
    # Make Predictions
    global y_pred
    y_pred = best_model.predict(x_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100.0}%")


    # Calcuate gruop-wise accuracy
    X_test_original = data.loc[x_test_indices]
    racial_groups = X_test_original['race'].unique()

    for group in racial_groups:
        group_indices = X_test_original[X_test_original['race'] == group].index
        group_y_test = y_test.loc[group_indices]
        group_y_pred = pd.Series(y_pred, index=y_test.index).loc[group_indices]
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        print(f"Accuracy for {group}: {group_accuracy * 100.0}%")

def get_y_pred():
    return y_pred

def get_best_model():
    return best_model