import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

def predict(image):

  base_file_path = os.path.dirname(os.path.abspath(__file__))
  labelsPath = os.path.join(base_file_path, "yolo.names")
  LABELS  =open(labelsPath).read().strip().split("\n")

  weightsPath = os.path.join(base_file_path,"yolov3_custom_train_3000.weights")
  configPath = os.path.join(base_file_path,"yolov3_custom_train.cfg")
  
  net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
  ln = net.getLayerNames()
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

  LABELS = open(labelsPath).read().strip().split("\n")

  (H, W) = image.shape[:2]
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)# Initializing for getting box coordinates, confidences, classid boxes = []
  confidences = []
  classIDs = []
  threshold = 0.15

  for output in layerOutputs:
      for detection in output:
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]        if confidence > threshold:
              box = detection[0:4] * np.array([W, H, W, H])
              (centerX, centerY, width, height) = box.astype("int")           
              x = int(centerX - (width / 2))
              y = int(centerY - (height / 2))    
              boxes.append([x, y, int(width), int(height)])
              confidences.append(float(confidence))
              classIDs.append(classID)


  idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)


  if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])       
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        text = "{}".format(LABELS[classIDs[i]])
        cv2.putText(image, text, (x + w, y + h),                     
        cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
  return(image)        
       
