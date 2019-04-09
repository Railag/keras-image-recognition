import numpy as np
from keras.applications.xception import Xception, decode_predictions

from utils import prepare_image


def predict():
    model = Xception()
    image = prepare_image(299, 299)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('Xception predictions:', decoded_predictions[0])
    res = np.argmax(pred[0])