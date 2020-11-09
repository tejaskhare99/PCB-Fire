import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image


def predict(image,color_pix,frame3, frame2): 
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
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608),
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
			flag_new = 0
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			for xtest in range(x,x+h):
				if flag_new==0:
					for ytest in range(y,y+w):
						for tx in color_pix:
							temp_sub = tx #color pixel
							temp_test = [ytest,xtest] #test_pixel
							if(temp_sub == temp_test):
								flag_new=1
				else:
					pass
#             print((x,y),(w,h))
			if flag_new==1:        
#                 (x, y) = (boxes[i][0], boxes[i][1])
#                 (w, h) = (boxes[i][2], boxes[i][3])
				if (LABELS[classIDs[i]] == 'capacitor'):
					c += 1
					color = (0, 255, 0)
					cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
					text = "{}".format(LABELS[classIDs[i]])
					cv2.putText(frame2, "c", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				elif (LABELS[classIDs[i]] == 'resistor'):
					r += 1
					color = (0, 0, 255)
					cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
					text = "{}".format(LABELS[classIDs[i]])
					cv2.putText(frame2, "r", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				elif (LABELS[classIDs[i]] == 'inductor'):
					ind += 1
					color = (255, 0, 255)
					cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
					text = "{}".format(LABELS[classIDs[i]])
					cv2.putText(frame2, "in", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				elif (LABELS[classIDs[i]] == 'ic'):
					ic += 1
					color = (0, 255, 255)
					cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
					text = "{}".format(LABELS[classIDs[i]])
					cv2.putText(frame2, "ic", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			else:
				pass



	print("missing capacitor: ", c)
	print("missing resistor: ", r)
	print("missing inductor: ", ind)
	print("missing ic: ", ic)



	return frame2, boxes, LABELS, classIDs


def run():

	frame1 = cv2.imread('D:/image_350.jpg') 
	# frame = cv2.imread('D:/Dataset/Assembled_PCB/26.png') 
	frame2 = cv2.imread('D:/image_350c3.jpg') 


	# # print(frame1.shape)
	if frame1.shape[0] > 1000:
		scale_percent = 25

	else:
		scale_percent = 100

	width = int(frame1.shape[1] * scale_percent / 100)
	height = int(frame1.shape[0] * scale_percent / 100)
	frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)
	
	# img1, box1, l1, id1 = predictonly(frame1)
	# cv2.imshow('output', img1)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	
	
	frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)
	frame3 = cv2.subtract(frame1,frame2)
	# cv2.imshow('output', frame3)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	
	f1 = cv2.imread('D:/image_350.jpg', 2) 
	f2 = cv2.imread('D:/image_350c3.jpg', 2) 
	if f1.shape[0] > 1000:
		scale_percent = 25

	else:
		scale_percent = 100

	width = int(f1.shape[1] * scale_percent / 100)
	height = int(f1.shape[0] * scale_percent / 100)
	f1 = cv2.resize(f1, (width, height), interpolation = cv2.INTER_AREA)
	f2 = cv2.resize(f2, (width, height), interpolation = cv2.INTER_AREA)
	f3 = cv2.subtract(f1,f2)
	# cv2.imshow('subtractbinary', f3)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	ret, bw_img = cv2.threshold(f3,127,255,cv2.THRESH_BINARY)
	# print("_____----------______", bw_img[2][1])
	color_pix = []
	for x in range(bw_img.shape[0]):
		for y in range(bw_img.shape[1]):
			temp = bw_img[x][y]
			if temp==255:
				color_pix.append([x,y])
				
	
	final, box2, l2, id2 = predict(frame1,color_pix,frame3, frame2)

	# print(color_pix)
	
	cv2.imshow('Detected Fault', final)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

c = time.time()
run()
print("Timeeeeeeee: ", time.time() - c)
