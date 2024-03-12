import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
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


def saveFltrImg(fltr, fl, cols):
	rows = math.ceil(len(fltr)/cols)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(fltr)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(fltr[index], cmap='gray')
	
	plt.savefig(fl)
	plt.close()

def LMFilter(scales, orientations, fltr_size):
	derivatives_scale = scales[0:3]
	gaussian_scale = scales
	LoG_scale = scales + [i * 3 for i in scales]

	fltr_bank = []
	first_derivatives = []
	second_derivatives = []
	gaussian = []
	LoG = []
	
    #sobel kernals
	Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sy = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])
	
    #18 first derivative filters and 18 second derivative filters are generated (# of scales = 3 * # of orientations = 6)
	for scale in derivatives_scale:
		sgm = [3*scale, scale]
		G = G2d(sgm, fltr_size) #2D elliptical gaussian with elongation factor of 3
		
		first_derivative = cv2.filter2D(G, -1, Sx) + cv2.filter2D(G, -1, Sy) #convolved gaussian with sobel
		second_derivative = cv2.filter2D(first_derivative, -1, Sx) + cv2.filter2D(first_derivative, -1, Sy) #convolved first derivative with sobel
		
        #rotating first and second derivatives to get desired orientations
		for orientation in range(orientations):
			fltr_orientation = orientation * 180 / orientations
     	
			first_derivative =  imutils.rotate(first_derivative, fltr_orientation)
			first_derivatives.append(first_derivative)

			second_derivative = imutils.rotate(second_derivative, fltr_orientation)
			second_derivatives.append(second_derivative)
	
	# 8 such filters are generated
	for scale in LoG_scale:
		sgm = sgm = [scale, scale]
		G = G2d(sgm, fltr_size)
		laplacian_kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		LoG.append(cv2.filter2D(G, -1, laplacian_kernal)) #convolving laplacian kernel and gaussian2D to get Laplacian of Gaussian filters
		
    # 4 gaussian filters are generated
	for scale in gaussian_scale:
		sgm = [scale, scale]
		gaussian.append(G2d(sgm, fltr_size))
		
    # total of 18 + 18 + 8 + 4 = 48 filters are generated
	fltr_bank = first_derivatives + second_derivatives + LoG + gaussian
	return fltr_bank

def main():
	folder_name = './Phase1/'
	LMS_fltr_bank = LMFilter([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49) # input = LMsmall filter scales
	saveFltrImg(LMS_fltr_bank,folder_name + "results/LMS.png", 12)
	LML_fltr_bank = LMFilter([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49) # input = LMlarge filter scales
	saveFltrImg(LML_fltr_bank, folder_name + "results/LML.png", 12)
	
if __name__ == '__main__':
    main()