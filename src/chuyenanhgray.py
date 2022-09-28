import cv2
import numpy as np
from PIL import Image,ImageTk
def chuyenanhgray(filehinh,congthuc):
    #Đọc ảnh màu dùng thư viện PIL. Ảnh PIL này chúng ta sẽ dùng thể thực hiện các tác vụ xử lý và tính toán thay vì dùng OpenCV
    imgPIL = Image.fromarray(filehinh)
    #Tạo một ảnh có cùng kích thước và mode với ảnh imgPIL. Ảnh này dùng để chứa kết quả chuyển đổi RGB sang Grayscale
    average = Image.new(imgPIL.mode,imgPIL.size)
    lightness=Image.new(imgPIL.mode,imgPIL.size)
    luminance=Image.new(imgPIL.mode,imgPIL.size)
    #Lấy kích thước của ảnh từ imgPIL
    width=average.size[0]
    height=average.size[1]
    #Mỗi ảnh là một ma trận 2 chiều nên sẽ dùng 2 vòng for để đọc hết các điểm ảnh có trong ảnh
    for x in range(width):
        for y in range(height):
            #Lấy giá trị điểm ảnh tại vị trí x,y
            R,G,B=imgPIL.getpixel((x,y))
            #Công thức chuyển đổi điểm ảnh màu RGB thành điểm ảnh mức xám dùng phương pháp Average
            if congthuc == 'average':
                gray_average=np.uint8((R+G+B)/3)
                # Gán giá trị mức xám vừa tính cho ảnh xám
                average.putpixel((x, y), (gray_average, gray_average, gray_average))
                # Chuyển ảnh từ PIL sang OpenCV
                anhavg = np.array(average)
                return anhavg
            #Công thức chuyển đổi điểm ảnh màu RGB thành điểm ảnh mức xám dùng phương pháp Lightness
            if congthuc == 'lightness':
                MIN=min(R,G,B)
                MAX=max(R,G,B)
                gray_lightness=np.uint8((MAX+MIN)/2)
                lightness.putpixel((x, y), (gray_lightness, gray_lightness, gray_lightness))
                anhlightness = np.array(lightness)
                return anhlightness
            #Công thức chuyển đổi điểm ảnh màu RGB thành điểm ảnh mức xám dùng phương pháp Luminance
            if congthuc == 'luminance':
                gray_luminance=np.uint8(0.2126*R+0.7152*G+0.0722*B )
                luminance.putpixel((x,y),(gray_luminance,gray_luminance,gray_luminance))
                anhluminance = np.array(luminance)
                return anhluminance

