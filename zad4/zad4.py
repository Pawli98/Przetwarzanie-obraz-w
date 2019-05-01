import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc
img = mc.imread('litery_1.png')

def rgb2gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    s1 = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return s1


def ffilter(img, mask):
	outputImg = np.copy(img)#Zwraca kopię tablicy danego obiektu.
	maskX,maskY=mask.shape#ustawianie tablicy
	midX = np.floor(maskX/2)
	midY = np.floor(maskY/2)#zaokrądlanie w dół
	
	lewydol = int(maskX-midX) #X lewy
	prawydol = int(maskX-lewydol) #X prawy
	gora = int(maskY-midY) #Y gora
	dol = int(maskY-gora) #Y dol

#jednoznaczność 
	tmp_image_with_offsets = np.zeros((int(img.shape[0] + gora + dol),int(img.shape[1] + prawydol + lewydol)))#przesuniecie
	for i in range(gora, tmp_image_with_offsets.shape[0] - dol):
		 for j in range(lewydol, tmp_image_with_offsets.shape[1] - prawydol):
 			tmp_image_with_offsets[i][j] = img[i - gora - dol][j - prawydol - lewydol]

	N=np.sum(np.sum(mask))#Funkcja sum () dodaje początek i elementy danej iteracji od lewej do prawej
	if(N==0):
		N=1
#szum-
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			part=tmp_image_with_offsets[i:i+maskX,j:j+maskY]
			outputImg[i,j]=np.sum(np.sum(np.multiply(part,mask)))/N
	return outputImg

def printFilter(img):
		
		#filtr Robertsa
		#dokladnosc :dobra
		#jednoznacznosc :	widac dobrze gdzie sa krawedzie
		#odpornosc na szum : wysoka
		mask = np.array([[-1,0],[1,0]])
		mask2 = np.array([[-1,1],[0,0]])
		mask3 = np.array([[0,1],[-1,0]])
		mask4 = np.array([[1,0],[0,-1]])
		test = np.abs(ffilter(img, mask))
		test2 = np.abs(ffilter(img, mask2))
		test3 = np.abs(ffilter(img, mask3))
		test4 = np.abs(ffilter(img, mask4))
		
		test = (test+test2+test3+test4)/4#zwracanie wartosci bezwzględnej
		plt.subplot(2, 3, 1)
 		#plt.imshow(img, cmap=plt.cm.gray , vmin=0 , vmax=255)
		
		plt.subplot(2, 3, 2)
		plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=255)
		
		#filtr Laplaca
		#dokladnosc : doklada
		#jednoznacznosc : krawedzie nie sa jednozaczne
		#odpornosc na szum : wysoka
        
		mask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
		mask2 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
		test = np.abs(ffilter(img, mask))
		test2 = np.abs(ffilter(img, mask2))
		test = (test+test2)/2
		plt.subplot(2, 3, 3)
		plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=255)
		
		# filtr Prewitta
		#dokladnosc : srednia
		#jednoznacznosc : sa jendoznaczne
		#odpornosc na szum: szum nie jest duzy
		mask = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
		mask2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
		mask3 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
		mask4 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
		test = np.abs(ffilter(img, mask))
		test2 = np.abs(ffilter(img, mask2))
		test3 = np.abs(ffilter(img, mask3))
		test4 = np.abs(ffilter(img, mask4))
		test = (test+test2+test3+test4)/4
		plt.subplot(2, 3, 4)
        
		plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=255)

		#filtr Sobela
		#dokladnosc: srednia
		#jednoznacznosc
		#odpornosc: duzy szum
		mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
		mask2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
		mask3 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
		mask4 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
		test = np.abs(ffilter(img, mask))
		test2 = np.abs(ffilter(img, mask2))
		test3 = np.abs(ffilter(img, mask3))
		test4 = np.abs(ffilter(img, mask4))
		test = (test+test2+test3+test4)/4
		plt.subplot(2, 3, 5)
		plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=255)
 		
		#filtr Kirischa
 		#dokladnosc : 		wszystkie krawedzie sa widoczne
 		#jednoznacznosc :	sa jednoznaczne
 		#odpornosc szum : duzy szum
		mask = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
		mask2 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
		mask3 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
		mask4 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
		mask5= np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
		mask6= np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
		mask7= np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
		mask8= np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
		test =   np.abs(ffilter(img, mask))
		test2 = np.abs(ffilter(img, mask2))
		test3 = np.abs(ffilter(img, mask3))
		test4 = np.abs(ffilter(img, mask4))
		test5 = np.abs(ffilter(img, mask5))
		test6 = np.abs(ffilter(img, mask6))
		test7 = np.abs(ffilter(img, mask7))
		test8 = np.abs(ffilter(img, mask8))
		test = (test+test2+test3+test4+test5+test6+test7+test8)/8
		plt.subplot(2, 3, 6)
       
		plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=255) 		
  		#plt.show();

img = rgb2gray(img)
printFilter(img)



