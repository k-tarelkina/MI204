import numpy as np
import cv2

from matplotlib import pyplot as plt


def normalize(image):
    return (image - np.min(image)) / np.ptp(image)


img = np.float64(cv2.imread("../Image_Pairs/FlowerGarden2.png", 0))

# 1. Calculate Sobel components using OpenCV

sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

sobel_module = np.sqrt(np.square(sobelx) + np.square(sobely))

# 2. Calculate "by hand"

x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobelx_manual = cv2.filter2D(img, -1, x_kernel)

y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobely_manual = cv2.filter2D(img, -1, y_kernel)

sobel_module_manual = np.sqrt(np.square(sobelx) + np.square(sobely))

# Plot results (images are normalized before plotting, e.g. brought to the range [0,1])

plt.figure(figsize=(6, 9))

plt.subplot(4, 2, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 3), plt.imshow(normalize(sobelx), cmap="gray")
plt.title("Sobel X (OpenCV)"), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 4), plt.imshow(normalize(sobelx_manual), cmap="gray")
plt.title("Sobel X"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 5), plt.imshow(normalize(sobely), cmap="gray")
plt.title("Sobel Y (OpenCV)"), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 6), plt.imshow(normalize(sobely_manual), cmap="gray")
plt.title("Sobel Y"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 7), plt.imshow(normalize(sobel_module), cmap="gray")
plt.title("Sobel Magnitude (OpenCV)"), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 8), plt.imshow(normalize(sobel_module_manual), cmap="gray")
plt.title("Sobel Magnitude"), plt.xticks([]), plt.yticks([])

plt.savefig("sobel.png")
plt.show()
