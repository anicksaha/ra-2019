import numpy as np
import keras
from keras.models import save_model, load_model
from keras.preprocessing.image import img_to_array
import cv2

if __name__ == "__main__":
  model = load_model("./emotion_detector_models/model_v6_23.hdf5")

  face_image = cv2.imread('newgirl_jess_.jpg')
  gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
  roi = gray.astype("float") / 255.0
  roi = img_to_array(roi)
  roi = np.expand_dims(roi, axis=0)
  predicted_class = np.argmax(model.predict(roi))
  class_labels=['angry','Disgust','Fear','Happy','Neutral','Sad','Suprised']
  print(class_labels[predicted_class])
