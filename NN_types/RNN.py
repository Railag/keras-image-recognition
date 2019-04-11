from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential
from numpy import array

from constants import n_features, n_steps
from utils import split_sequence


def fit_model(X, y):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=200, verbose=0)

    return model


def predict(training_input, test_input):
    X, y = split_sequence(training_input, n_steps)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = fit_model(X, y)

    x_input = array(test_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    pred = model.predict(x_input, verbose=0)
    print("RNN prediction: " + str(pred))
