import numpy as np
from keras.applications.mobilenet import MobileNet, decode_predictions

from utils import prepare_image


def predict():
    model = MobileNet()
    image = prepare_image(224, 224)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('MobileNet predictions:', decoded_predictions[0])
    np.argmax(pred[0])
