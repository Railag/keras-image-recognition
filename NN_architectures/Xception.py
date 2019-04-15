import numpy as np
from keras.applications.xception import Xception, decode_predictions


def predict(image):
    model = Xception()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('Xception predictions:', decoded_predictions[0])
    res = np.argmax(pred[0])
    return decoded_predictions[0]

