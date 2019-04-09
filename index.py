import os

from NN_architectures.InceptionV3 import predict as InceptionV3_predict
from NN_architectures.MobileNet import predict as MobileNet_predict
from NN_architectures.MobileNetV2 import predict as MobileNetV2_predict
from NN_architectures.ResNet import predict as ResNet50_predict
from NN_architectures.VGG16 import predict as VGG16_predict
from NN_architectures.VGG19 import predict as VGG19_predict
from NN_architectures.Xception import predict as Xception_predict
###
from NN_types.CNNSample import predict as CNN_predict
from NN_types.LSTMSample import predict as LSTM_predict
from utils import prepare_image

# hiding tensorflow logs
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def time_series(training_input, test_input):
    print("Input: " + str(test_input))
    LSTM_predict(training_input, test_input)
    CNN_predict(training_input, test_input)
    # TODO more NN types predictions


def image_feature_recognition(image_path):
    image224 = prepare_image(image_path, 224, 224)
    image299 = prepare_image(image_path, 299, 299)

    VGG16_predict(image224)
    VGG19_predict(image224)
    ResNet50_predict(image224)
    InceptionV3_predict(image299)
    Xception_predict(image299)
    MobileNet_predict(image224)
    MobileNetV2_predict(image224)


print("NN architectures")
image_feature_recognition('test.jpg')

print("\n\n")
print("NN types")
training_input_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90]
test_input_1 = [70, 80, 90]
time_series(training_input_1, test_input_1)

training_input_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70, 60, 50, 40]
test_input_2 = [90, 80, 100]
time_series(training_input_2, test_input_2)
