import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split




def model(input_shape,num_classes):
   
      # Build the network model
    model = Sequential() #API mô hình tuần tự

    #Tạo Convolutionnal Layers : Conv2D1 với số filter là 32, kích thước 3x3
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))   #hàm kích hoạt ReLu

    # Tạo Convolutionnal Layers : Conv2D2 với số filter là 64, kích thước 3x3
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization()) #chuẩn hóa dữ liệu ở các layer theo batch về phân phối chuẩn để quá trình gradient descent (suy giảm độ dốc) hội tụ nhanh hơn
    model.add(Activation("relu"))

    # Tạo Convolutionnal Layers : Conv2D3 với số filter là 64, kích thước 1x1
    model.add(Conv2D(64, (1, 1)))
    model.add(Dropout(0.5))     #giảm 50% số node
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Tạo Convolutionnal Layers : Conv2D4 với số filter là 64, kích thước 3x3
    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    # Tạo pooling Layers : kích thước 2x2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Tạo Convolutionnal Layers : Conv2D5 với số filter là 64, kích thước 1x1
    model.add(Conv2D(64, (1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(32))    #layer ẩn với 32 neural
    model.add(Dense(num_classes))   #layer đầu ra là số lượng neural bằng đầu vào
    model.add(Activation("softmax"))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',                              #Stochastic gradient descent
              metrics=['accuracy'])     #đánh giá mô hình
              
    model.summary() #Bảng đánh giá model

    return model
    
