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
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Activation, Flatten
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from Model import model
from keras import callbacks
import time
start = time.time()
# Đường dẫn đến hình ảnh database
path = 'FaceReg/dataset'
# path = 'dataset'

# recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L') #Chuyển đổi kiểu dữ liệu float32 sang uint8  và tạo Pillow Image object bằng cách load data từ numpy
    img = img.resize((32,32), Image.ANTIALIAS)   #resize thành 32x32 và khử răng cưa bằng Image.ANTIALIAS
    return np.array(img)



#hàm lấy hình ảnh và dữ liệu label
def getImagesAndLabels(path):
    
    # path = 'FaceReg/dataset'
    # path = 'dataset'

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        
        #try/except khi có lỗi trong lúc lấy ảnh
        try:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        except:
            continue    
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces now.")
faces,ids = getImagesAndLabels(path)

K.clear_session() # giải phóng trạng thái toàn cục: giúp tránh sự lộn xộn từ các mô hình và lớp cũ. Giải phóng bộ nhớ
n_faces = len(set(ids))
model = model((32,32,1),n_faces)
faces = np.asarray(faces)
faces = np.array([downsample_image(ab) for ab in faces])
ids = np.asarray(ids)
faces = faces[:,:,:,np.newaxis] #np.newaxis: tăng kích thước của mảng hiện có thêm một chiều
print("Shape of Data: " + str(faces.shape))
print("Number of unique faces : " + str(n_faces))


ids = to_categorical(ids) #mã hóa one_hot ids

faces = faces.astype('float32')
faces /= 255. # normalize từ 0-255 về 0-1

#Tách tập dữ liệu train và test với đầu vào là faces và đầu ra là ids
x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.2, random_state = 0)
#test size: phần dữ liệu được sử dụng để kiểm tra là 20%=0.2


# kiểm tra, lưu lại các trạng thái và thông số của model
checkpoint = tf.keras.callbacks.ModelCheckpoint('FaceReg/dataset/trained_model.h5', monitor='val_accuracy',
                                       save_best_only=True, save_weights_only=True, verbose=1)
#monitor: tham số quan sát
#save_best_only: model chỉ lưu lại một checkpoint tốt nhất
#save_weights_only: callback sẽ chỉ lưu lại weights, không lưu lại cấu trúc model


#bắt đầu training dữ liệu
model.fit(x_train, y_train,     #x_train: input data, #y_train: target data
             batch_size=32,     #số lượng mẫu mỗi lần cập nhật
             epochs=60,
             validation_data=(x_test, y_test),  #dữ liệu để đánh giá số liệu sau mỗi chu kỳ epoch
             shuffle=True,callbacks=[checkpoint])   #xáo trộn dữ liệu sau mỗi chu kỳ epoch
             
end = time.time()
# Thời gian huấn luyện
print("[INFO] training took {:.4f} seconds".format(end-start))
# In số lượng khuôn mặt được huấn luyện
print("\n [INFO] " + str(n_faces) + " faces trained. Exiting Program")
