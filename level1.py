import json
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import csv
 
 
img1=cv2.imread("Level_1_Input_Data/wafer_image_1.png")
img2=cv2.imread("Level_1_Input_Data/wafer_image_2.png")
img3=cv2.imread("Level_1_Input_Data/wafer_image_3.png")
img4=cv2.imread("Level_1_Input_Data/wafer_image_4.png")
img5=cv2.imread("Level_1_Input_Data/wafer_image_5.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5=cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
images=[img1,img2,img3,img4,img5]

 
 
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff
error, diff = mse(img1, img2)
print("Image matching Error between the two images:",error)

cv2.imshow("difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
 

result=[]
for i in range(0,600):
    for j in range(0,800):
        l=[img1[i][j],img2[i][j],img3[i][j],img4[i][j],img5[i][j]]
        s=set(l)
        c=[]        
        if len(s)>1 and len(s)<5:          
           for n in s:
              e=l.count(n)
              c.append(e)
           
          # m = min(c)
           
          #ind=c.index(m)
          
          # s=list(s)
           m = max(c)
           ind=c.index(m)
           s=list(s)
        
           my_list = np.array(l) 
           indices = np.where(my_list != s[ind])[0]
           
           for k in indices:
               print(k+1,j,abs(i-599))
               lt=[k+1,j,abs(i-599)]
               result.append(lt)
         
      
with open ('defectdie3.csv','w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(result)

df=pd.read_csv('defectdie3.csv',header=None,index_col=None)
print(df)