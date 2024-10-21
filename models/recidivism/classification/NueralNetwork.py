import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

class NueralNetwork:
    def __init__(self, input_dim):
        self.base_model = Sequential()
        self.best_model = None
        self.y_pred = None
        self.base_model.add(Dense(32, input_dim=2, activation='relu'))
        self.base_model.add(Dense(1, activation='sigmoid'))
        self.base_model.compile(loss='binary_crossentropy', 
                                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                                metrics=['accuracy'])
        self.model = KerasClassifier(build_fn=lambda: self.base_model, epochs=10,
                                     batch_size=32, verbose=0)

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