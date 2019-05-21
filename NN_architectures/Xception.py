import numpy as np
from keras.applications.xception import Xception, decode_predictions


def predict(image):
    model = Xception()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'Xception predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

