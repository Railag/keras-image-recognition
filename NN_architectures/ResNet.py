import numpy as np
from keras.applications.resnet50 import ResNet50, decode_predictions


def predict(image):
    model = ResNet50()

    pred = model.predict(image)
    decoded_predictions = decode_predictions(pred, top=10)
    response = 'ResNet50 predictions:   ' + str(decoded_predictions[0][0:5])
    print(response)
    np.argmax(pred[0])
    return response

