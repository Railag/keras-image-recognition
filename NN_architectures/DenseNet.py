import numpy as np
from keras.applications.densenet import DenseNet201, decode_predictions


def predict(image):
    model = DenseNet201()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'DenseNet201 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

