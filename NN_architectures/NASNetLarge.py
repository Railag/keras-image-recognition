import numpy as np
from keras.applications.nasnet import NASNetLarge, decode_predictions


def predict(image):
    model = NASNetLarge()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('NASNetLarge predictions:', decoded_predictions[0])
    np.argmax(pred[0])
    return decoded_predictions[0]

