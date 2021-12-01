#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:19:58 2021

@author: caojiajia
"""
# In[資料填寫]
Number = input("請輸入員工代號：")
print("臉部辨識檔案將儲存為", Number +'.jpg')
facefile = Number + ".jpg"

# In[1] #照相環節(第一次進入建立資料集)

from PIL import Image
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #臉部偵測模組

cv2.namedWindow("Photo")
cap = cv2.VideoCapture(0)  
                    
while(cap.isOpened()):                          # 如果攝影機有開啟就執行迴圈
    ret, img = cap.read()                       # 讀取影像
    cv2.imshow("Photo", img)                    # 顯示影像在OpenCV視窗
    if ret == True:                             # 讀取影像如果成功
        key = cv2.waitKey(200)                  # 等一下攝影機
        if key == ord("1") :  
            cv2.imwrite( Number +".jpg", img)       # 按1拍照存檔           
            break
cap.release()                                   # 關閉攝影機ㄅ

cv2.waitKey(3000)
cv2.destroyAllWindows()


faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))
# 標註右下角底色是黃色
cv2.rectangle(img, (img.shape[1]-120, img.shape[0]-20),
              (img.shape[1],img.shape[0]), (0,255,255), -1)
# 標註找到多少的人臉
cv2.putText(img, "Find " + str(len(faces)) + " face",
            (img.shape[1]-110, img.shape[0]-5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

# 將人臉框起來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      # 藍色框住人臉
    myimg = Image.open(Number +".jpg")                     # PIL模組開啟
    imgCrop = myimg.crop((x, y, x+w, y+h))              # 裁切
    imgResize = imgCrop.resize((150,150), Image.ANTIALIAS)
    imgResize.save( Number +"facefile.jpg")                       # 儲存檔案
    
cv2.namedWindow("FaceRecognition", cv2.WINDOW_NORMAL)
cv2.imshow("FaceRecognition", img)

cv2.waitKey(3000)
cv2.destroyAllWindows()
# In[2] 之後進入機房時拍照偵測臉部對比

cv2.namedWindow("Photo")
cap = cv2.VideoCapture(0)  
                    
while(cap.isOpened()):                          # 如果攝影機有開啟就執行迴圈
    ret, img = cap.read()                       # 讀取影像
    cv2.imshow("Photo", img)                    # 顯示影像在OpenCV視窗
    if ret == True:                             # 讀取影像如果成功
        key = cv2.waitKey(200)                  # 等一下攝影機
        if key == ord("1") :  
            cv2.imwrite( "faceout.jpg", img)       # 按1拍照存檔           
            break
cap.release()                                   # 關閉攝影機ㄅ

cv2.waitKey(3000)
cv2.destroyAllWindows()


faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
        minNeighbors = 3, minSize=(20,20))
# 標註右下角底色是黃色
cv2.rectangle(img, (img.shape[1]-120, img.shape[0]-20),
              (img.shape[1],img.shape[0]), (0,255,255), -1)
# 標註找到多少的人臉
cv2.putText(img, "Find " + str(len(faces)) + " face",
            (img.shape[1]-110, img.shape[0]-5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

# 將人臉框起來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      # 藍色框住人臉
    myimg = Image.open("faceout.jpg")                     # PIL模組開啟
    imgCrop = myimg.crop((x, y, x+w, y+h))              # 裁切
    imgResize = imgCrop.resize((150,150), Image.ANTIALIAS)
    imgResize.save("faceoutCut.jpg")                       # 儲存檔案
    

cv2.waitKey(3000)
cv2.destroyAllWindows()
# In[臉型比對資料]
from functools import reduce
from PIL import Image
import math
import operator

h1 = Image.open( Number +"facefile.jpg").histogram()
h2 = Image.open("faceoutCut.jpg").histogram()

RMS = math.sqrt(reduce(operator.add, list(map(lambda a,b:
                                              (a-b)**2 , h1 ,h2)))/len(h1))
print("RMS值：" , RMS)  #RMS越低代表臉部偵測越符合

if RMS <=100:
    print("審核通過，請進入機房")
else:
    print("比對失敗，請通知主管人員處理")
    
#RMS值： 49.688325464841334
#審核通過，請進入機房
