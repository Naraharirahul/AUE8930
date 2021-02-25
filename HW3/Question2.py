import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
img = mpimg.imread('Images/Lenna.jpg')  

# 2.1
# Using the NTSC approach 
grayimg = 0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]

bins = np.zeros(256)
for i in range(256):
    for j in range(256):
        a = int(grayimg[i,j])
        bins[a] = bins[a] + 1
bins_a = np.arange(0,256,1)
# plt.plot(bins_a,bins)
# plt.xlabel('Pixel intensity')
# plt.ylabel('Pixel frequency')
# plt.show()

# 2.2

acumm_bins = np.zeros(256)
acumm_bins[0] = bins[0]
for i in range(0,256):
    acumm_bins[i] = acumm_bins[i - 1] + bins[i]

acumm_bins_a = np.arange(0,256,1)
# plt.plot(acumm_bins_a,acumm_bins)
# plt.xlabel(' Pixel intensity')
# plt.ylabel('Cummulative Pixel frequency')
# plt.show()

# 2.3

normalized_bins = bins / (grayimg.shape[0] * grayimg.shape[1])
acc_normalized_bins = np.zeros(256)
acc_normalized_bins[0] = normalized_bins[0]
for i in range(256):
    acc_normalized_bins[i] = normalized_bins[i] + acc_normalized_bins[i-1]

grayimg_equalized = np.zeros((256,256))

for i in range(256):
    for j in range(256):
        grayimg_equalized[i,j] = acc_normalized_bins[int(grayimg[i,j])] * (256-1)

plt.imshow(grayimg_equalized,cmap='gray')
plt.show()

new_bins = np.zeros(256)
for i in range(256):
    for j in range(256):
        a = int(grayimg_equalized[i,j])
        new_bins[a] = new_bins[a] + 1

new_bins_acc = np.zeros(256)
new_bins_acc[0] = new_bins[0]
for i in range(0,256):
    new_bins_acc[i] = new_bins[i] + new_bins_acc[i-1]

new_bins_a = np.arange(0,256,1)
plt.plot(new_bins_a,new_bins)
plt.xlabel('Pixel intensity')
plt.ylabel('Accumulative Pixel frequency ')
plt.show()