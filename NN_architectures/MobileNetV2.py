import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions


def predict(image):
    model = MobileNetV2()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'MobileNetV2 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

