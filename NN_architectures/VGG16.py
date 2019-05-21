import numpy as np
from keras.applications.vgg16 import VGG16, decode_predictions


def predict(image):
    model = VGG16()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'VGG16 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response
