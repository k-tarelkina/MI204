import numpy as np
import cv2

from matplotlib import pyplot as plt

# Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread("../Image_Pairs/Graffiti0.png", cv2.IMREAD_GRAYSCALE))
(h, w) = img.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")
print("Type de l'image :", img.dtype)

# Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

derivative_x = cv2.Sobel(Theta, cv2.CV_64F, dx=1, dy=0, ksize=3)
derivative_y = cv2.Sobel(Theta, cv2.CV_64F, dx=0, dy=1, ksize=3)

derivative_xx = np.square(derivative_x)
derivative_xy = np.multiply(derivative_x, derivative_y)
derivative_yy = np.square(derivative_y)

# Here Gaussian filter is used. One can also use mean with kernel as np.ones((3, 3))
kernel = np.random.normal(1, 1, (3, 3))

derivative_xx = cv2.filter2D(derivative_xx, -1, kernel)
derivative_xy = cv2.filter2D(derivative_xy, -1, kernel)
derivative_yy = cv2.filter2D(derivative_yy, -1, kernel)

det_M = np.multiply(derivative_xx, derivative_yy) - np.square(derivative_xy)
trace_M = derivative_xx + derivative_yy

# A typical value for k is in the range of 0.04 to 0.06
k = 0.04
Theta = det_M - k * np.square(trace_M)

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc, d_maxloc), np.uint8)
Theta_dil = cv2.dilate(Theta, se)

# Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0

# On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif * Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Mon calcul des points de Harris :", time, "s")
print("Nombre de cycles par pixel :", (t2 - t1) / (h * w), "cpp")

plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title("Image originale")

plt.subplot(132)
plt.imshow(Theta, cmap="gray")
plt.title("Fonction de Harris")

se_croix = np.uint8(
    [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ]
)
Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)

# Relecture image pour affichage couleur

Img_pts = cv2.imread("../Image_Pairs/Graffiti0.png", cv2.IMREAD_COLOR)
(h, w, c) = Img_pts.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")
print("Type de l'image :", Img_pts.dtype)

# On affiche les points (croix) en rouge

Img_pts[Theta_ml_dil > 0] = [255, 0, 0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title("Points de Harris")

plt.show()
