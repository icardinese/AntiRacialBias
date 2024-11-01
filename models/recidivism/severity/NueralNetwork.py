import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

class NueralNetwork:
    def __init__(self, input_dim, num_classes=9):  # num_classes for multi-class classification
        self.base_model = Sequential()
        self.base_model.add(Dense(32, input_dim=input_dim, activation='relu'))
        # Output layer with num_classes (9 for severity levels 0-8)
        self.base_model.add(Dense(num_classes, activation='softmax'))
        
        # Compile for multi-class classification
        self.base_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                metrics=['accuracy'])
        
        self.model = KerasClassifier(build_fn=lambda: self.base_model, epochs=10, batch_size=32, verbose=0)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        self.y_pred = self.model.predict(x_test)
        return self.y_pred

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_y_pred(self):
        return self.y_pred

    def get_model(self):
        return self.model
    
    # Save the model
    def save(self, filename):
        self.base_model.save(f'neural_network_model_{filename}.h5')
    
    # Load the model
    def load(self, filename):
        self.base_model = tf.keras.models.load_model(f"neural_network_model_{filename}.h5")
        # Re-wrap the model with KerasClassifier if necessary
        self.model = KerasClassifier(build_fn=lambda: self.base_model, epochs=10,
                                     batch_size=32, verbose=0)