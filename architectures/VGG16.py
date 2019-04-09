import numpy as np
from keras.applications.vgg16 import VGG16, decode_predictions

from utils import prepare_image


def predict():
    model = VGG16()
    image = prepare_image(224, 224)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('VGG16 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
