from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

import numpy as np
import os
import glob

subfolder = ['../data/faces/']
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
    mappings.update({idx:img_path})
    idx+=1


embeddings_list_np = np.array(embeddings_list)
np.savetxt('embeddings.txt', embeddings_list_np)

# Save
np.save('mappings.npy', mappings) 
