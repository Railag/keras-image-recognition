import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions

from utils import prepare_image


def predict():
    model = MobileNetV2()
    image = prepare_image(224, 224)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('MobileNetV2 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
