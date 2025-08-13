import cv2 as cv
import pandas as pd
import numpy as np
import ultralytics as yolo
import torchvision as tv
#from roboflow import Roboflow

# loading pre-trained model


'''
TODO:

 - train pytorch model for YOLO class to use
 - train model using annotated hands images


 - using different sized bounding boxes to determine what task the submarine will do first (?)
 - inverse square law- there are different light intensities within the pool from sunlight vs shaded environments, won't work

 - 
'''


trained_model_data = "../pytorch/best.pt"
test_trained_model_data = "../yolov8n.pt"
model = yolo.YOLO(test_trained_model_data)
capture = cv.VideoCapture(0)


training_results = model.train(data="..RoboFlow/data.yaml", epochs=100)

while True:
    ret, frame = capture.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    cv.imshow("Detect Stuff", annotated_frame)
    if cv.waitKey(1) and 0xFF == ord("q"):
        break
    
capture.release()
cv.destroyAllWindows()


# able to use video, directory, URL, etc.
#results = model.predict(source="image.jpg") 


#results[0].show() # show results of first image 
