import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import numpy as np
import PIL
from PIL import ImageOps
import numpy
from IPython.display import Image, display

u2unet_model = keras.models.load_model('./ett/10-08-4000-u2netModel')
attention_model = keras.models.load_model('./ett/8-08-2000-Attention_unetModel')
r2unet_model = keras.models.load_model('./ett/10-10-4000-r2unetModel')
vnet_model = keras.models.load_model('./ett/0714-2000-VnetModel')

#將圖做前處理(CLAHE和ROI)
##CLAHE
def clahe(img):
  clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (12,12))
  clahe_image = clahe.apply(img)
  return clahe_image

##ROI
def roi(img, img_height, img_width):
  x = img_width//2
  y = int(img_height*0.1)
  roi_img = img[y:y+1024, x-512:x+512]
  return roi_img

##define DataSet
def define_dataset(fileName):
  ettImgPath = './media/'+fileName
  img = cv2.imread(ettImgPath, cv2.IMREAD_GRAYSCALE)

  Predict_num = 1
  predict_data = np.zeros((Predict_num,) + (512,512) + (1,), dtype="float32")
  img_height = img.shape[0]
  img_width = img.shape[1]
  # generate CLAHE
  ettImg = clahe(img)
  # # generate ROI
  ettImg = roi(ettImg, img_height, img_width)
  ettImg = cv2.resize(ettImg, (512, 512), interpolation=cv2.INTER_AREA)
  predict_data[0] = np.expand_dims(ettImg/255, 2)

  #predict
  ETT_result_1 = r2unet_model.predict(predict_data,batch_size=1)
  ETT_result_2 = attention_model.predict(predict_data,batch_size=1)
  ETT_result_3 = u2unet_model.predict(predict_data,batch_size=1)
  ETT_result_4 = vnet_model.predict(predict_data,batch_size=1)

  return ETT_result_1,ETT_result_2,ETT_result_3,ETT_result_4,ettImg

#處理預測結果並轉換格式成uint8
##convertImageToNumpy
def ConverResultToCv2(i,result):
    mask = np.argmax(result[i], axis=-1)
    mask = np.array(mask,np.uint8)
    for i in range(mask.shape[0]):
      for j in range(mask.shape[1]):
        if(mask[i][j] != 0):
          mask[i][j] = 255
    return mask
def ConverResultToCv2_Unet3p_U2net(i,result):
    mask = np.argmax(result[0][i], axis=-1) 
    mask = np.array(mask,np.uint8)
    for i in range(mask.shape[0]):
      for j in range(mask.shape[1]):
        if(mask[i][j] != 0):
          mask[i][j] = 255
    return mask

#印出預測圖
# def display_mask_test(ETT_result_1,fileName):
#     mask = np.argmax(ETT_result_1[0], axis=-1)#swinUNet
#     mask = np.array(mask,np.uint8)
#     for i in range(mask.shape[0]):
#       for j in range(mask.shape[1]):
#         if(mask[i][j] != 0):
#           mask[i][j] = 255
#     # cv2.imshow('',mask)
#     # cv2.waitKey(0)
#       cv2.imwrite("./ett/static/upload/new_"+fileName,mask)

# 系集
def ensemble_ett(ETT_result_1,ETT_result_2,ETT_result_3,ETT_result_4):
    i=0
    img1 = ConverResultToCv2(i,ETT_result_1)
    img2 = ConverResultToCv2(i,ETT_result_2)
    img3 = ConverResultToCv2_Unet3p_U2net(i,ETT_result_3)
    img4 = ConverResultToCv2(i,ETT_result_4)

    newImg1 = cv2.addWeighted(img1,0.5,img2,0.5,0)
    newImg2 = cv2.addWeighted(img3,0.5,img4,0.5,0)
    newImg3 = cv2.addWeighted(newImg1,0.5,newImg2,0.5,0)

    for x in range(newImg3.shape[0]):
      for y in range(newImg3.shape[1]):
        px = newImg3[x,y]
        if(px>=200):
          newImg3[x,y] = 255
        else:
          newImg3[x,y] = 0
    resultImg = newImg3
    return resultImg
 

#findContours(氣管內管端點、carina端點)
def findContours(img):
  try:
    # cv2.findContours: 找到ROI的輪廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 用一個list來存每個區塊的最低點
    LowerPoints = []
    # contours有很多區塊的輪廓，這邊把所有區都跑過一次
    for i in contours:
      # 經過下面這段code會找到該區塊最低點，並存起來
      LowerPoints.append(tuple(i[i[:,:,1].argmax()][0]))
    # 這邊會排序所有區塊的最低點
    LowerPoints = sorted(LowerPoints,key=lambda t:t[1],reverse=True)
    #描輪廓
    # 再取得y值最大的那一筆(第0筆)
    Botton_Point = LowerPoints[0]
    return Botton_Point
  except:
    print('找不到點')

#印出標註點 ett圖

def ETT_PrintPoint(resultImg,ettImg,fileName):
  ##取得ett最低點
  PredictBottonPoints = []
  for i in range(1):
    PredictBottonPoints.append(findContours(resultImg))

  ##畫點上去，底圖:原圖
  for i in range(1):
    ettImg = cv2.cvtColor(ettImg, cv2.COLOR_BGR2RGB)
    cv2.circle(ettImg, PredictBottonPoints[i], 3, (0,0,255), 3)
    #儲存
    cv2.imwrite("./ett/static/upload/new_"+fileName,ettImg)