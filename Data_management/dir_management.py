import os
import glob
from shutil import copyfile

os.chdir('/home/vaibhav/tejas research/data/') #where all the separate folders are
main_dir = os.listdir()
i=1

for folder in main_dir:
    os.chdir('/home/vaibhav/tejas research/data/'+folder+'/')
    files = glob.glob("*.jpg")
    for z in files:
        copyfile(z,'/home/vaibhav/tejas research/images/image_'+str(i)+'.jpg')
        i=i+1
            
print(i)        
