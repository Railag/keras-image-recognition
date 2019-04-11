from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array

from constants import n_features, n_steps
from utils import split_sequence, visualize_history


def fit_model(X, y):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['acc', 'mse', 'mae', 'mape', 'cosine'])

    history = model.fit(X, y, validation_split=0.25, epochs=200, verbose=1)

    # visualize_history(history)

    return model


def predict(training_input, test_input):
    X, y = split_sequence(training_input, n_steps)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = fit_model(X, y)

    x_input = array(test_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    pred = model.predict(x_input, verbose=0)
    print("LSTM prediction: " + str(pred))
