import numpy as np
#fft傅里叶装换， ifft傅里叶反装换
from numpy.fft import fft, ifft
from PIL import Image

gou = Image.open("F:\python项目\自然语言处理\gou.jpg")
#gou.show()

gou_data = np.fromstring(gou.tobytes(), dtype=np.int8) #0-255 int8最大是128(超过就是负数)， int16=128*128太大
#print(gou_data)#[-70 -52 -37 ... 126  76  -1]
#print(len(gou_data)) #print(len(gou_data))

#将真是的数据装换成 - 频域
gou_fft = fft(gou_data)
#[29904280.               +0.j          6864831.1978689 -27740460.7281791j
#  2592712.30156177+16641319.50530075j ...
# -5547041.56230945 -1912417.86034378j  2592712.30156172-16641319.50530076j
#  6864831.197869  +27740460.72817908j]
#  j虚数
#print(gou_fft)


#滤波---
#去掉低频的波---低频代表变化不大,高频是陡变 --轮廓
#np.where(np.abs(gou_fft) < 1e10, 0, gou_fft) #0替换小于1e5（100000）data的绝对值 的值

gou_fft[np.where(np.abs(gou_fft) < 1e5)] = 0


#傅里叶反装
gou_ifft = ifft(gou_fft)
#print(gou_ifft)
gou_data_ifft = np.real(gou_ifft) #获取实数
#print(gou_data_ifft)
gou_1 = np.int8(gou_data_ifft)
#print(gou_1)

gou_jpg = Image.frombytes(data=gou_1, size=gou.size, mode=gou.mode)
gou_jpg.show()