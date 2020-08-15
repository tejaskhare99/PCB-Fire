import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

def predict(image,color_pix,frame3): 
    # base_file_path = os.path.dirname(os.path.abspath(__file__))
    labelsPath = "/home/vaibhav/tejas research/predict/yolo.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath =  "/home/vaibhav/tejas research/weights/yolov3_custom_train_4000.weights"
    configPath = "/home/vaibhav/tejas research/predict/yolov3_custom_train.cfg"

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
            for xtest in range(x,x+h+1):
                if flag_new==0:
                    for ytest in range(y,y+w+1):
                        for ztest in range(3):
                            for tx in color_pix:
                                temp_sub = tx #color pixel
                                temp_test = [xtest,ytest,ztest] #test_pixel
                                if(temp_sub == temp_test):
                                    flag_new=1
                else:
                    pass

            if flag_new==1:        
#                 (x, y) = (boxes[i][0], boxes[i][1])
#                 (w, h) = (boxes[i][2], boxes[i][3])
                if (LABELS[classIDs[i]] == 'capacitor'):
                    c += 1
                    color = (0, 255, 0)
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 1)
                    text = "{}".format(LABELS[classIDs[i]])
                    cv2.putText(frame3, "c", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif (LABELS[classIDs[i]] == 'resistor'):
                    r += 1
                    color = (0, 0, 255)
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 1)
                    text = "{}".format(LABELS[classIDs[i]])
                    cv2.putText(frame3, "r", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif (LABELS[classIDs[i]] == 'inductor'):
                    ind += 1
                    color = (255, 0, 255)
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 1)
                    text = "{}".format(LABELS[classIDs[i]])
                    cv2.putText(frame3, "in", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif (LABELS[classIDs[i]] == 'ic'):
                    ic += 1
                    color = (0, 255, 255)
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 1)
                    text = "{}".format(LABELS[classIDs[i]])
                    cv2.putText(frame3, "ic", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                pass



    print("capacitor: ", c)
    print("resistor: ", r)
    print("inductor: ", ind)
    print("ic: ", ic)



    return frame3, boxes, LABELS, classIDs


def run():

    frame1 = cv2.imread('/home/vaibhav/tejas research/subtract test/image_350.jpg') 
    # frame = cv2.imread('D:/Dataset/Assembled_PCB/26.png') 
    frame2 = cv2.imread('/home/vaibhav/tejas research/subtract test/image_350c.jpg') 
    # # print(frame1.shape)
    if frame1.shape[0] > 1000:
        scale_percent = 25

    else:
        scale_percent = 100

    width = int(frame1.shape[1] * scale_percent / 100)
    height = int(frame1.shape[0] * scale_percent / 100)
    frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)
    frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)
    frame3 = cv2.subtract(frame1,frame2)
    color_pix = []
    for x in range(frame3.shape[0]):
        for y in range(frame3.shape[1]):
            for z in range(frame3.shape[2]):
                if frame3[x][y][z]!=0:
                    color_pix.append([x,y,z])
# 	img1, box1, l1, id1 = predict(frame1)
    final, box2, l2, id2 = predict(frame1,color_pix,frame3)

# # 	for A in range(len(box1)):
# # 	    flag.append('not_found')


# # 	for x in range(len(box1)):
# # 	    for y in range(len(box2)):
# # 	        a_ar = box1[x]
# # 	        b_ar = box2[y]
# # 	        a_xc = a_ar[0]
# # 	        a_yc = a_ar[1]
# # 	        a_hc = a_ar[2]
# # 	        a_wc = a_ar[3]

# # 	        b_xc = b_ar[0]
# # 	        b_yc = b_ar[1]
# # 	        b_hc = b_ar[2]
# # 	        b_wc = b_ar[3]

# # 	        if a_xc-5<= b_xc <= a_xc+5:
# # 	            if a_yc-5<= b_yc <= a_yc+5:
# # 	                if a_hc-5<= b_hc <= a_hc+5:
# # 	                    if a_wc-5<= b_wc <= a_wc+5:
# # 	                        flag[x]='found'

# # 	        else:
# # 	            pass


# 	for l in range(len(flag)):
# 	    if(flag[l]=='not_found'):
# 	#         print('element number {} is not found'.format(l+1))
# 	        co.append(box1[l])
# 	        label.append(l1[id1[l]])
# 	print("CO: ", co)
# 	print(label)
# 	# print(id1)

# 	frame2 = cv2.imread('D:/image_350c.jpg') 
# 	frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)					
# 	for i in range(len(co)):
# 		(x, y) = (co[i][0], co[i][1])
# 		(w, h) = (co[i][2], co[i][3])  
# 		color = (255,255,0)	    
# 		cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 1)
# 		cv2.putText(frame2, "Missing " + label[i], (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

    cv2.imshow('output', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()	

#     plt.imshow(img)
#     plt.plot()

run()
