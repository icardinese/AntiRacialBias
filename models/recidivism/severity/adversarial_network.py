import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

class AdversarialNetwork:
    def __init__(self, input_dim, num_classes=9):  # num_classes for multi-class classification
        # Define the main model (meta-classifier)
        self.main_model = Sequential()
        self.main_model.add(Dense(32, input_dim=input_dim, activation='relu'))
        # Multi-class output layer
        self.main_model.add(Dense(num_classes, activation='softmax'))
        
        # Compile for multi-class classification
        self.main_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define the adversary model (debiasing component)
        self.adversary_model = Sequential()
        self.adversary_model.add(Dense(16, input_dim=1, activation='relu'))
        self.adversary_model.add(Dense(1, activation='sigmoid'))
        self.adversary_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Wrap the main model in KerasClassifier (from scikeras)
        self.model = KerasClassifier(model=self.main_model, epochs=10, batch_size=32, verbose=0)

    def fit(self, X_train, y_train, race_train, batch_size=32):
        # Train the main model (meta-classifier)
        self.model.fit(X_train, y_train, batch_size=batch_size)

        # Get predictions from the main model
        main_pred = self.model.predict(X_train).ravel()

        # Train the adversary model (debiasing model)
        self.adversary_model.fit(main_pred, race_train, epochs=1, batch_size=batch_size, verbose=0)

    def predict(self, X_test):
        # Predict using the main model
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        # Predict probabilities using the main model
        return self.model.predict_proba(X_test)

    def get_model(self):
        # Return the main model
        return self.model
