import numpy as np
from keras.applications.nasnet import NASNetMobile, decode_predictions


def predict(image):
    model = NASNetMobile()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('NASNetMobile predictions:', decoded_predictions[0])
    np.argmax(pred[0])
    return decoded_predictions[0]

