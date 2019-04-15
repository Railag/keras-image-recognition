import numpy as np
from keras.applications.vgg19 import VGG19, decode_predictions


def predict(image):
    model = VGG19()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('VGG19 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
    return decoded_predictions[0]
