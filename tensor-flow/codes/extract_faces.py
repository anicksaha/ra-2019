import os
import glob
import cv2

imagenames_list = []
subfolder = ['../data/anne','../data/emma']

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
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    #cv2.imshow('Image',roi_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    file_name = 'faces_' + str(idx) + '.jpg'
    write_file_path = '../data/faces/' + file_name
    cv2.imwrite(write_file_path, roi_color)
    idx+=1
