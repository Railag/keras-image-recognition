import numpy as np
from keras.applications.resnet50 import ResNet50, decode_predictions

from utils import prepare_image


def predict():
    model = ResNet50()
    image = prepare_image(224, 224)

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('ResNet50 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
