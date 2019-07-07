from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
#from extract_faces import extract_images
from sklearn.cluster import KMeans

import numpy as np
import os
import glob
import pylab as plt

subfolder = ['./data/faces/']
imagenames_list=[]
#subfolder = ['./data/anne','./data/emma']
for idx,dirname in enumerate(subfolder):
    for f in glob.glob(dirname+'/*.jpg'):
        imagenames_list.append(f)

print(imagenames_list)



model = ResNet50(weights='imagenet', include_top=False)
# model.summary()

vgg_feature_list = []
    
  
for file_path in imagenames_list:
    img_path = file_path
    #print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg_feature = model.predict(img_data)
    vgg_feature_np = np.array(vgg_feature)
    normalized_reduced_examples = np.nan_to_num((vgg_feature_np - np.mean(vgg_feature_np, axis=0)) / np.std(vgg_feature_np, axis=0))
    vgg_feature_list.append(normalized_reduced_examples.flatten())
    



vgg_feature_list_np = np.array(vgg_feature_list)
# test_list=[]
# subfolder1 = ['./data/test']
# for idx,dirname in enumerate(subfolder1):
#     for f in glob.glob(dirname+'/*.jpg'):
#         test_list.append(f)
# vgg_feature_test_list = []
# for file_path in test_list:
#     img_path = file_path
#     print(img_path)
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data, axis=0)
#     img_data = preprocess_input(img_data)
#     vgg_feature = model.predict(img_data)
#     vgg_feature_np = np.array(vgg_feature)
#     vgg_feature_test_list.append(vgg_feature_np.flatten())


from sklearn.neighbors import NearestNeighbors

n_neighbors = 5
nnbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(vgg_feature_list_np)
distances, neighbors_indices = nnbrs.kneighbors(vgg_feature_list_np)
print(neighbors_indices)

scores = 1 - (distances / np.max(distances))
#print(scores)
#Since the first neighbor is itself, the first column will correspond to input image
n_rows = 1
normalized_reduced_examples_visualized_indices = np.random.randint(low=0, high=20, size=(n_rows))
embedding = np.empty((n_rows * n_neighbors, 2))
embedding_filenames = [None] * (n_rows * n_neighbors)

for row in range(n_rows):
    example_index = normalized_reduced_examples_visualized_indices[row]
    example_neighbors = neighbors_indices[example_index]
    for column in range(n_neighbors):
        i = row * n_neighbors + column
        embedding[i, 0] = column
        embedding[i, 1] = row
        embedding_filenames[i] = imagenames_list[example_neighbors[column]]

print(embedding_filenames)
#check if weights are proper
#randomized index of image use map
#gray scale
#open face, VGG face net50