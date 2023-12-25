from PIL import Image

import numpy as np
from numpy import asarray
from numpy import expand_dims

from keras_facenet import FaceNet

import pickle
import cv2
from test import test

import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

cap = cv2.VideoCapture(0)


while(1):
    _, gbr1 = cap.read()
    

    label = test(
                image=gbr1,
                model_dir='resources/anti_spoof_models',
                device_id=0
                )
    
    if label == 1:
        wajah = HaarCascade.detectMultiScale(gbr1,1.1,4)

        if len(wajah)>0:
            x1, y1, width, height = wajah[0]        
        else:
            x1, y1, width, height = 1, 1, 10, 10
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        
        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
        gbr_array = asarray(gbr)
        
        face = gbr_array[y1:y2, x1:x2] 

        face = Image.fromarray(face)                       
        face = face.resize((160,160))
        face = asarray(face)
        

        
        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)
        
        min_dist=100
        # min_dist=1

        identity=' '
        for key, value in database.items():
            dist = np.linalg.norm(value - signature)

            if dist < min_dist:
                min_dist = dist
                if dist <= 1:
                    identity = key
                else:
                    identity = "tidak dikenali"
                
        cv2.putText(gbr1,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(gbr1,(x1,y1),(x2,y2), (0,255,0), 2)
        additional_text_position = (100, 130)  # Adjust the Y-coordinate as needed
        cv2.putText(gbr1,  str(min_dist), additional_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            
    # Add label to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gbr1, str(label), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('label', gbr1)


    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
