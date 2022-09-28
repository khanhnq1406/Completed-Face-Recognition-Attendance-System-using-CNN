import cv2
import os
from newattendace import *
from subprocess import call

cam = cv2.VideoCapture(0)

#Dùng thuật toán xác định khuôn mặt haarcascade_frontalface
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w+50,y+h+50), (255,0,0), 2)     
        count += 1 #sau mỗi lần lấy ảnh thành công, biến count sẽ tăng 1

        # Lưu hình ảnh đã chụp vào folder "dataset"
        gray = gray[y:y+h,x:x+w]

        #Lưu ảnh với định dạng "User.id.count.jpg
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg",gray )

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Nhấn 'ESC' để thoát khỏi quá trì lấy dataset
    if k == 27:
        break
    elif count >= 100: # Chụp 70 tấm ảnh là mẫu khuôn mặt và dừng
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")


cam.release()
cv2.destroyAllWindows()


