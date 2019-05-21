import numpy as np
from keras.applications.nasnet import NASNetMobile, decode_predictions


def predict(image):
    model = NASNetMobile()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'NASNetMobile predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

