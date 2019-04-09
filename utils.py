from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img


def prepare_image(x, y):
    image = load_img('test.jpg', target_size=(x, y))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image
