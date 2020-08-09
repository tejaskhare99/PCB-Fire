import cv2

frame1 = cv2.imread('/home/vaibhav/tejas research/subtract test/image_350.jpg')
frame2 = cv2.imread('/home/vaibhav/tejas research/subtract test/image_350c.jpg')                  
if frame1.shape[0] > 1000:
    scale_percent = 25
else:
    scale_percent = 100
width = int(frame1.shape[1] * scale_percent / 100)
height = int(frame1.shape[0] * scale_percent / 100)
A = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)
B = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)
C = cv2.subtract(A,B)

color_pix = []

for x in range(C.shape[0]):
    for y in range(C.shape[1]):
        for z in range(C.shape[2]):
            test.append((x,y,z))
            if C[x][y][z]!=0:
                color_pix.append([x,y,z])



def box_remove(idxs,color_pix,C):
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
                        for ztest in range(z):
                            for tx in range color_pix:
                                temp_sub = tx
                                temp_test = [xtest,ytest,ztest]
                                if(temp_sub == temp_test):
                                    flag_new=1
                else:
                    pass
                                
            if flag_new==1:        
          
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
                
            else:
                pass
    
