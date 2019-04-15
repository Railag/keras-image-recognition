import numpy as np
from keras.applications.densenet import DenseNet201, decode_predictions


def predict(image):
    model = DenseNet201()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    print('DenseNet201 predictions:', decoded_predictions[0])
    np.argmax(pred[0])
    return decoded_predictions[0]

