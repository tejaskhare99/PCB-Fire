import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

def predict(image): 
	# base_file_path = os.path.dirname(os.path.abspath(__file__))
	labelsPath = "D:/darknet/data/yolo.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	weightsPath =  "D:/darknet/yolov3_custom_train_4000.weights"
	configPath = "D:/darknet/cfg/yolov3_custom_train.cfg"

	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	LABELS = open(labelsPath).read().strip().split("\n")

	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		   swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)# Initializing for getting box coordinates, confidences, classid 
	boxes = []
	confidences = []
	classIDs = []
	threshold = 0.15

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]        
			if confidence > threshold:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")           
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))    
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)


	if len(idxs) > 0:
		c = 0
		r = 0
		ind = 0
		ic = 0
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])   

			if (LABELS[classIDs[i]] == 'capacitor'):
				c += 1
				color = (0, 255, 0)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(image, "c", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			elif (LABELS[classIDs[i]] == 'resistor'):
				r += 1
				color = (0, 0, 255)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(image, "r", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			elif (LABELS[classIDs[i]] == 'inductor'):
				ind += 1
				color = (255, 0, 255)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(image, "in", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			elif (LABELS[classIDs[i]] == 'ic'):
				ic += 1
				color = (0, 255, 255)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(image, "ic", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		print("capacitor: ", c)
		print("resistor: ", r)
		print("inductor: ", ind)
		print("ic: ", ic)



							
	return image, boxes, LABELS, classIDs



def run():

	frame1 = cv2.imread('D:/image_537.jpg') 
	# frame = cv2.imread('D:/Dataset/Assembled_PCB/26.png') 
	frame2 = cv2.imread('D:/image_537c.jpg') 
	# print(frame1.shape)
	if frame1.shape[0] > 1000:
		scale_percent = 25
		
	else:
		scale_percent = 100
	
	width = int(frame1.shape[1] * scale_percent / 100)
	height = int(frame1.shape[0] * scale_percent / 100)
	frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)
	frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)
	img1, box1, l1, id1 = predict(frame1)
	img2, box2, l2, id2 = predict(frame2)
	print("1: ", np.asarray(box1))
	print("\n2: ", np.asarray(box2))
	c = 1
	co = []
	label = []
	for i in range(len(box1)): 
		for j in range(len(box2)): 
			if (box1[i] != box2[j]): 
				c += 1
				if c == len(box2):
					print("Mis ", box1[i])
					co.append(box1[i])
					label.append(l1[id1[i]])
	print("CO: ", co)
	print(label)
	# print(id1)
	cv2.imshow('missing', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
						
	for i in range(len(co)):
		(x, y) = (co[i][0], co[i][1])
		(w, h) = (co[i][2], co[i][3])  
		color = (0,169,255)	    
		cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 1)
		cv2.putText(frame2, "Missing " + label[i], (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

	cv2.imshow('output', img1)
	cv2.imshow('missing', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

#     plt.imshow(img)
#     plt.plot()

run()







# x = np.asarray([[1, 2, 3, 4], [22, 4, 1, 4]])
# x = np.pad(x, [(0, 1), (0, 0)], mode='constant')
# print(x)
