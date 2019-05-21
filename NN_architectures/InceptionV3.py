import numpy as np
from keras.applications.inception_v3 import InceptionV3, decode_predictions


def predict(image):
    model = InceptionV3()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'InceptionV3 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

