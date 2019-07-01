import numpy as np
import os
import glob
import cv2

def extract_images(subfolder):

    imagenames_list=[]

    for idx,dirname in enumerate(subfolder):
        for f in glob.glob(dirname+'/*.jpg'):
            imagenames_list.append(f)

    #print(imagenames_list)

    face_cascade = cv2.CascadeClassifier('../face-detection/data/haarcascades/haarcascade_frontalface_default.xml')

    i = 1

    for file_path in imagenames_list:
        img_path = file_path
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        #cv2.imshow('Image',roi_color)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        file_name = 'faces_' + str(i) + '.jpg'
        write_file_path = './data/faces/' + file_name
        cv2.imwrite(write_file_path, roi_color)
        i=i+1
    
    
