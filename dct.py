import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

img = cv2.imread('bear.jpeg')
img = cv2.resize(img,(1400,880))
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV_I420)###轉YUV
img1 = img.astype('float')
C_temp = np.zeros((8,8),float)
dst = np.zeros((8,8),float)
 
yuv_d_1 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
yuv_r_1 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
yuv_d_4 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
yuv_r_4 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
yuv_d_16 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
yuv_r_16 = np.zeros((yuv.shape[0],yuv.shape[1]), float)
m = 8
n = 8
N = n
C_temp[0, :] = 1 * np.sqrt(1/N)
 
for i in range(1, m):
     for j in range(n):
          C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )) * np.sqrt(2 / N )
#print(C_temp)

width_d =175
height_d =165
width = 1400
height = 880

for i in range(height_d):
    for j in range(width_d):
        
        dst = np.dot(C_temp , yuv[i*8:(i+1)*8,j*8:(j+1)*8])
        dst = np.dot(dst, np.transpose(C_temp))
    
        yuv_d_1[i*8,j*8] = dst[0,0] 
        yuv_d_4[i*8:i*8+2,j*8:j*8+2] = dst[0:2,0:2]
        yuv_d_16[i*8:i*8+4,j*8:j*8+4] = dst[0:4,0:4]
        
        dst = np.dot(np.transpose(C_temp),yuv_d_1[i*8:(i+1)*8,j*8:(j+1)*8])
        dst = np.dot(dst, C_temp )
        
        yuv_r_1[i*8:(i+1)*8,j*8:(j+1)*8] = dst[:,:]
        
        dst = np.dot(np.transpose(C_temp),yuv_d_4[i*8:(i+1)*8,j*8:(j+1)*8])
        dst = np.dot(dst, C_temp )
        
        yuv_r_4[i*8:(i+1)*8,j*8:(j+1)*8] = dst[:,:]

        dst = np.dot(np.transpose(C_temp),yuv_d_16[i*8:(i+1)*8,j*8:(j+1)*8])
        dst = np.dot(dst, C_temp )
        
        yuv_r_16[i*8:(i+1)*8,j*8:(j+1)*8] = dst[:,:]

###for 1    
dst_i = yuv_r_1.astype('uint8')
dst_i = cv2.cvtColor(dst_i ,cv2.COLOR_YUV2BGR_I420)
cv2.imshow('1', dst_i)

mse_1=0
for i in range(880):
    for j in range(1400):
        mse_1 =mse_1 + (((yuv_r_1[i][j]-yuv[i][j]))**2)/(1400*880)

psnr_1= 10*np.log10((255**2)/mse_1)

###for 4
dst_i = yuv_r_4.astype('uint8')
dst_i = cv2.cvtColor(dst_i ,cv2.COLOR_YUV2BGR_I420)
cv2.imshow('4', dst_i)

mse_4=0
for i in range(880):
    for j in range(1400):
        mse_4 =mse_4 + (((yuv_r_4[i][j]-yuv[i][j]))**2)/(1400*880)

psnr_4= 10*np.log10((255**2)/mse_4)

###for 16
dst_i = yuv_r_16.astype('uint8')
dst_i = cv2.cvtColor(dst_i ,cv2.COLOR_YUV2BGR_I420)
cv2.imshow('16', dst_i)

mse_16=0
for i in range(880):
    for j in range(1400):
        mse_16 =mse_16 + ((yuv_r_16[i][j]-yuv[i][j])**2)/(1400*880)

psnr_16= 10*np.log10((255**2)/mse_16)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("PSNR 1 = ",round(psnr_1,2))
print("PSNR 4 = ",round(psnr_4,2))
print("PSNR 16 = ",round(psnr_16,2))

