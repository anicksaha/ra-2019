from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np

model = VGG19(weights='imagenet', include_top=False)
# model.summary()

img_path = './test/mila.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg_feature = model.predict(img_data)

# print(vgg_feature)
# print(vgg_feature.shape)








