from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def predict(training_input, test_input):
    n_steps = 3

    X, y = split_sequence(training_input, n_steps)

    n_features = 1

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=200, verbose=0)

    x_input = array(test_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    pred = model.predict(x_input, verbose=0)
    print("LSTM prediction: " + str(pred))
