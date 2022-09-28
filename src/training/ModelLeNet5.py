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
from keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization, AveragePooling2D
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split




def model(input_shape,num_classes):
   
      # Build the network model
    model = Sequential() #API mô hình tuần tự

    #Tạo Convolutionnal Layers : Conv2D1 với số filter là 32, kích thước 3x3
    model.add(Conv2D(6, (5, 5), input_shape=input_shape))
    model.add(Activation("sigmoid"))   #hàm kích hoạt ReLu

    # Tạo pooling Layers : kích thước 2x2
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Tạo Convolutionnal Layers : Conv2D1 với số filter là 32, kích thước 3x3
    model.add(Conv2D(16, (5, 5)))
    model.add(Activation("sigmoid"))  # hàm kích hoạt ReLu

    # Tạo pooling Layers : kích thước 2x2
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Tạo Convolutionnal Layers : Conv2D1 với số filter là 32, kích thước 3x3
    model.add(Conv2D(120, (5, 5)))
    model.add(Activation("sigmoid"))  # hàm kích hoạt ReLu


    model.add(Flatten())
    model.add(Dense(84))    #layer ẩn với 32 neural
    model.add(Dense(num_classes))   #layer đầu ra là số lượng neural bằng đầu vào (ở đây là 2 người)
    model.add(Activation("softmax"))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
    model.summary() #Bảng đánh giá model

    return model
    
