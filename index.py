import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

model = VGG16()
image = load_img('test.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

pred = model.predict(image)
decodedPredictions = decode_predictions(pred, top=10)
print('Predicted:', decodedPredictions[0])
np.argmax(pred[0])

# TODO launch recognitions from different neural nets and compare results
