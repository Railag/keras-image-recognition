import numpy as np
from keras.applications.vgg19 import VGG19, decode_predictions

from utils import prepare_image


def predict():
    model = VGG19()
    image = prepare_image(224, 224)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('VGG19 predictions:', decoded_predictions[0])
    np.argmax(pred[0])