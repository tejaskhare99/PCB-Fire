import os
os.chdir('/home/vaibhav/tejas research/labels/label/') #label directory
a = os.listdir()
count = []
for x in a:
    file1 = open(x,"r") 
    w = file1.readlines()
    file1.close()
    for y in w:
        temp = y.split(" ")
        count.append(temp[0])
        
        
cap=0
res=0
ind=0
ic=0

for z in count:
    if z=='15':
        cap=cap+1
    elif z=='16':
        res=res+1
    elif z =='17':
        ind=ind+1
    elif z == '18':
        ic=ic+1
    else:
        pass
        

print("Number of capacitors are:",cap)
print("Number of resistors are:",res)
print("Number of inductors are:",ind)
print("Number of ic are:",ic)
        
        

