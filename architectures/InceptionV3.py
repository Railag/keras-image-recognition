import numpy as np
from keras.applications.inception_v3 import InceptionV3, decode_predictions

from utils import prepare_image


def predict():
    model = InceptionV3()
    image = prepare_image(299, 299)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('InceptionV3 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
