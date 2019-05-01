import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc

def normalizacja(count,image):
	offset=255.0/count
	output = np.zeros((image.shape[0],image.shape[0]))
	i = 0
	tmp = 0
	while (i < 255.0):
		for j in range (0,image.shape[0]):
			for k in range (0,image.shape[1]):
				if (image[j][k] > i and image[j][k] <= i+offset):
					output[j][k] = tmp
		tmp = tmp+1
		i=i+offset
	return output
	


def rgb2gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    s1 = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return s1

def test(img, dlx, dly):
	P = np.max(img).astype(int)+1
	C = np.zeros((P,P))
	wysokosc=len(img)
	szerokosc=len(img[0])
	
	for i in range (0,wysokosc-dlx):
		for j in range (0,szerokosc-dly):
			x=img[i,j]
			x=x.astype(int)
			y=img[i+dlx,j+dly]
			y=y.astype(int)
			C[x,y] = C[x,y]+1
	return C	

liczba_przedzialow = 255

img  = mc.imread('oko1.png')	
img = rgb2gray(img)
img = normalizacja(liczba_przedzialow, img)
img2  = mc.imread('oko2.png')	
img2 = rgb2gray(img2)
img2 = normalizacja(liczba_przedzialow, img2)



plt.subplot(2, 2, 1);
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=liczba_przedzialow)
plt.title("oko1.png")
plt.subplot(2, 2, 2);
plt.imshow(test(img, 0, 1), cmap=plt.cm.gray, vmin=0, vmax=liczba_przedzialow)
plt.title("macierz wspowwystepowania oko1.png")


plt.subplot(2, 2, 3);
plt.imshow(img2, cmap=plt.cm.gray, vmin=0, vmax=liczba_przedzialow)
plt.title("oko2.png")
plt.subplot(2, 2, 4);
plt.imshow(test(img2, 0, 1), cmap=plt.cm.gray, vmin=0, vmax=liczba_przedzialow)
plt.title("macierz wspowwystepowania oko2.png")

plt.show()