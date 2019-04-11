from keras.layers import Dense
from keras.models import Sequential
from numpy import array

from constants import n_steps
from utils import split_sequence


def fit_model(X, y):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=2000, verbose=0)

    return model


def predict(training_input, test_input):
    X, y = split_sequence(training_input, n_steps)

    model = fit_model(X, y)

    x_input = array(test_input)
    x_input = x_input.reshape((1, n_steps))

    pred = model.predict(x_input, verbose=0)
    print("MLP prediction: " + str(pred))
