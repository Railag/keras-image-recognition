from architectures.VGG16 import predict as VGG16_predict
from architectures.VGG19 import predict as VGG19_predict
from architectures.ResNet import predict as ResNet50_predict
from architectures.InceptionV3 import predict as InceptionV3_predict
from architectures.Xception import predict as Xception_predict
from architectures.MobileNet import predict as MobileNet_predict
from architectures.MobileNetV2 import predict as MobileNetV2_predict

# TODO launch recognitions from different neural nets and compare results


VGG16_predict()
VGG19_predict()
ResNet50_predict()
InceptionV3_predict()
Xception_predict()
MobileNet_predict()
MobileNetV2_predict()
