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

# TODO launch recognitions from different neural nets and compare results

print("NN architectures")
VGG16_predict()
VGG19_predict()
ResNet50_predict()
InceptionV3_predict()
Xception_predict()
MobileNet_predict()
MobileNetV2_predict()

print("\n\n")
print("NN types")
training_input = [10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70, 60, 50, 40]
test_input = [90, 80, 100]
print("Input: " + str(test_input))
LSTM_predict(training_input, test_input)
CNN_predict(training_input, test_input)
