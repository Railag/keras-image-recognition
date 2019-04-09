from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from numpy import array

from constants import n_features, n_steps
from utils import split_sequence


def fit_model(X, y):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=1000, verbose=0)

    return model


def predict(training_input, test_input):
    X, y = split_sequence(training_input, n_steps)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = fit_model(X, y)

    x_input = array(test_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    pred = model.predict(x_input, verbose=0)
    print("CNN prediction: " + str(pred))
