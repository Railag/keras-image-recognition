import numpy as np
from keras.applications.mobilenet import MobileNet, decode_predictions


def predict(image):
    model = MobileNet()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('MobileNet predictions:', decoded_predictions[0])
    np.argmax(pred[0])
    return decoded_predictions[0]

