import numpy as np
from keras.applications.vgg19 import VGG19, decode_predictions


def predict(image):
    model = VGG19()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'VGG19 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response
