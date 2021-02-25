import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
img = mpimg.imread('Images/Lenna.jpg')  

# 1.1 
# Using the NTSC approach 
grayimg = 0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]

plt.imshow(grayimg, cmap = 'gray')
plt.savefig('Images/LennaGray.jpg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

# 1.2

down_sampled_img = np.zeros(shape = (64,64))

for i in range(64):
    for j in range(64):
        down_sampled_img[i,j] = grayimg[i*4, j*4] 

plt.imshow(down_sampled_img,cmap='gray')
plt.show()

#1.3

sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 

sobel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 

sobel_filtered_image = np.zeros(shape = (256,256))
for i in range(256 - 2):
    for j in range(256 - 2):
        gx = np.sum(np.multiply(sobel_v, grayimg[i:i + 3, j:j + 3]))  # x direction
        
        gy = np.sum(np.multiply(sobel_h, grayimg[i:i + 3, j:j + 3]))  # y direction
        
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  


plt.imshow(sobel_filtered_image, cmap = 'gray')
plt.show()