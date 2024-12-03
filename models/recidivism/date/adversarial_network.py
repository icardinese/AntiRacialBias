import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

class AdversarialNetwork:
    def __init__(self, input_dim):  # Regression model, no num_classes
        # Define the main model (meta-regressor)
        self.main_model = Sequential()
        self.main_model.add(Dense(32, input_dim=input_dim, activation='relu'))
        # Output layer with a single unit for regression
        self.main_model.add(Dense(1, activation='linear'))
        
        # Compile for regression
        self.main_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        # Define the adversary model (debiasing component)
        self.adversary_model = Sequential()
        self.adversary_model.add(Dense(16, input_dim=1, activation='relu'))
        self.adversary_model.add(Dense(1, activation='sigmoid'))
        self.adversary_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Wrap the main model in KerasRegressor (from scikeras)
        self.model = KerasRegressor(model=self.main_model, epochs=10, batch_size=32, verbose=0)

    def fit(self, X_train, y_train, race_train, batch_size=32):
        # Train the main model (meta-regressor)
        self.model.fit(X_train, y_train, batch_size=batch_size)

        # Get predictions from the main model
        main_pred = self.model.predict(X_train).ravel()

        # Train the adversary model (debiasing model)
        self.adversary_model.fit(main_pred, race_train, epochs=1, batch_size=batch_size, verbose=0)

    def predict(self, X_test):
        # Predict using the main model (meta-regressor)
        return self.model.predict(X_test)

    def get_model(self):
        # Return the main model
        return self.model
    
    # Save the model
    def save(self, filename):
        self.main_model.save(f'advmainneural_network_model_{filename}.h5')
        self.adversary_model.save(f'adversaryneural_network_model_{filename}.h5')
    
    # Load the model
    def load(self, filename):
        self.model = tf.keras.models.load_model(f"advmainneural_network_model_{filename}.h5")
        # Re-wrap the model with KerasClassifier if necessary
        self.adversary_model = tf.keras.models.load_model(f"adversaryneural_network_model_{filename}.h5")
        return self.model
