from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from sklearn.cluster import KMeans

import numpy as np
import os
import glob
import pylab as plt

subfolder = ['./data/anne','./data/emma']
imagenames_list=[]

for idx,dirname in enumerate(subfolder):
    for f in glob.glob(dirname+'/*.jpg'):
        imagenames_list.append(f)

print(imagenames_list)

model = VGG19(weights='imagenet', include_top=False)
# model.summary()

vgg_feature_list = []

for file_path in imagenames_list:
    img_path = file_path
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg_feature = model.predict(img_data)
    vgg_feature_np = np.array(vgg_feature)
    vgg_feature_list.append(vgg_feature_np.flatten())


vgg_feature_list_np = np.array(vgg_feature_list)

# kmeans = KMeans(n_clusters = 2, random_state = 0).fit(vgg_feature_list_np)
