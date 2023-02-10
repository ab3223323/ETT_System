# ETT_System
分析氣管內管端點與隆突端點  
輸入胸腔x光圖，印出預測結果及分析距離  
分析多種model並系集  
![image](https://user-images.githubusercontent.com/80948966/210208686-d547c19f-90df-47ca-a337-e6cac4e45780.png)
![image](https://user-images.githubusercontent.com/80948966/210208695-f31d1dbd-ef13-43cc-92a3-07394662fe06.png)

## 資料集
使用RANZCR CLiP - Catheter and Line Position Challenge 約3000張胸腔x光  
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview  

## Import  
```
import tensorflow as tf 
from tensorflow import keras
import cv2
import pandas as pd
import numpy as np
```
## 將圖做前處理(CLAHE和ROI)
```
def clahe(img):
  clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (12,12))
  clahe_image = clahe.apply(img)
  #output image
  return clahe_image
```
```
def roi(img, img_height, img_width):
  x = img_width//2
  y = int(img_height*0.1)
  roi_img = img[y:y+1024, x-512:x+512]

  # cv2_imshow(roi_img)
  return roi_img

```
