import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

class NueralNetwork:
    def __init__(self, input_dim):  # Regression, so no num_classes
        self.base_model = Sequential()
        self.base_model.add(Dense(32, input_dim=input_dim, activation='relu'))
        # Single output for regression
        self.base_model.add(Dense(1, activation='linear'))
        
        # Compile for regression
        self.base_model.compile(loss='mean_squared_error',
                                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                metrics=['mean_squared_error'])
        
        # Wrap the model for regression
        self.model = KerasRegressor(build_fn=lambda: self.base_model, epochs=10, batch_size=32, verbose=0)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def get_model(self):
        return self.model
    
    # Save the model
    def save(self, filename):
        self.base_model.save(f'neural_network_model_{filename}.h5')
    
    # Load the model
    def load(self, filename):
        self.base_model = tf.keras.models.load_model(f"neural_network_model_{filename}.h5")
        # Re-wrap the model with KerasClassifier if necessary
        self.model = KerasRegressor(build_fn=lambda: self.base_model, epochs=10,
                                     batch_size=32, verbose=0)
