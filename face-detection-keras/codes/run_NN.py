import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings_list_np = np.loadtxt('embeddings.txt')
mappings = np.load('mappings.npy') 
n_neighbors = 5
nnbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(embeddings_list_np)
distances, neighbors_indices = nnbrs.kneighbors(embeddings_list_np)
print(neighbors_indices)


### Load test data and check on them

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

import numpy as np
import os
import glob
import cv2

#extracting faces from images
imagenames_list = []
subfolder = ['../data/test']

for idx,dirname in enumerate(subfolder):
    for file_path in glob.glob(dirname+'/*.jpg'):
        imagenames_list.append(file_path)

face_cascade = cv2.CascadeClassifier('../metadata/haarcascades/haarcascade_frontalface_default.xml')

idx = 1 ## face file naming

for file_path in imagenames_list:
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
    
    file_name = 'test_' + str(idx) + '.jpg'
    write_file_path = '../data/test_faces/' + file_name
    cv2.imwrite(write_file_path, roi_color)
    idx+=1

subfolder = ['../data/test_faces/']
imagenames_list=[]
for idx,dirname in enumerate(subfolder):
    for file_path in glob.glob(dirname+'/*.jpg'):
        imagenames_list.append(file_path)


model = ResNet50(weights='imagenet', include_top=False)

embeddings_list = []
mappings = {}

idx = 0 # embedding_list index
for img_path in imagenames_list:
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    embedding = model.predict(img_data)
    embedding_np = np.array(embedding)
    embeddings_list.append(embedding_np.flatten())
    idx+=1


test_list_np = np.array(embeddings_list)
distances, neighbors_indices = nnbrs.kneighbors(test_list_np)
print(neighbors_indices)