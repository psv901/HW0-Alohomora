import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#Gaussian2D function
def G2d(sgm, sz):	
	sgm_x, sgm_y = sgm
	if (sz%2) == 0:
		idx = sz/2
	else:
		idx = (sz - 1)/2

	x, y = np.meshgrid(np.linspace(-idx, idx+1, sz), np.linspace(-idx, idx+1, sz))
	p = (np.square(x)/np.square(sgm_x)) + (np.square(y)/np.square(sgm_y))
	gaussian = (1/(2 * np.pi * sgm_x * sgm_y)) * np.exp(-(p/2))
	return gaussian

#Derivative of Gaussian Filters
def DoGFilters(orientations, scales, fltr_sz):

	fltr_bank = []
	#sobel kernals
	Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])  
    
	
	for scale in scales:
		sgm = [scale, scale]
		G = G2d(sgm, fltr_sz)

		Gx = cv2.filter2D(G,-1, Sx)
		Gy = cv2.filter2D(G,-1, Sy)
		for orient in range(orientations):
			fltr_orientation = orient * 2 * np.pi / orientations 
			fltr = (Gx * np.cos(fltr_orientation)) +  (Gy * np.sin(fltr_orientation))
			fltr_bank.append(fltr)
        
		
	return fltr_bank


def saveFltrImg(fltr, fl, cols):
	rows = math.ceil(len(fltr)/cols)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(fltr)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(fltr[index], cmap='gray')
	
	plt.savefig(fl)
	plt.close()


def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	folder_name = "./Phase1/"

	dog_fltr_bank = DoGFilters(16, [3,5], 50)
	saveFltrImg(dog_fltr_bank, folder_name + "results/DoG.png", cols = 8)



if __name__ == '__main__':
    main()