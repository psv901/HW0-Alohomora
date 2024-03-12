#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import math
import scipy
import scipy.stats as st
import skimage.transform
import sklearn.cluster
import os

#Gaussian2D function
def G2d(sgm, sz):	
	sgm_x, sgm_y = sgm
	if (sz%2) == 0:
		idx = sz/2
	else:
		idx = (sz - 1)/2

	x, y = np.meshgrid(np.linspace(-idx, idx+1, sz), np.linspace(-idx, idx+1, sz))

	# implementing gaussian 2D formula 
	p = (np.square(x)/np.square(sgm_x)) + (np.square(y)/np.square(sgm_y))
	gaussian = (1/(2 * np.pi * sgm_x * sgm_y)) * np.exp(-(p/2))
	return gaussian

def sin2d(frequency, sz, theta):
	if (sz%2) == 0:
		idx = sz/2
	else:
		idx = (sz - 1)/2

	x, y = np.meshgrid(np.linspace(-idx, idx+1, sz), np.linspace(-idx, idx+1, sz))
	z = x * np.cos(theta) + y * np.sin(theta)
	sin2d = np.sin(z * 2 * np.pi * frequency/sz)

	return sin2d


#Derivative of Gaussian Filters
def DoGFilters(orientations, scales, fltr_sz):

	fltr_bank = []
	#sobel kernals
	Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])  
    
	
	for scale in scales:
		sgm = [scale, scale]
		G = G2d(sgm, fltr_sz)

		# convolving gaussian and sobel 
		Gx = cv2.filter2D(G,-1, Sx) 
		Gy = cv2.filter2D(G,-1, Sy)

		# rotating the filters based on the number of orientations given as ipnut
		for o in range(orientations):
			fltr_orientation = o * 2 * np.pi / orientations 
			fltr = (Gx * np.cos(fltr_orientation)) +  (Gy * np.sin(fltr_orientation))
			fltr_bank.append(fltr)
        
	return fltr_bank


def LMFilter(scales, orientations, fltr_sz):
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
		G = G2d(sgm, fltr_sz) #2D elliptical gaussian with elongation factor of 3
		
		first_derivative = cv2.filter2D(G, -1, Sx) + cv2.filter2D(G, -1, Sy) #convolved gaussian with sobel
		second_derivative = cv2.filter2D(first_derivative, -1, Sx) + cv2.filter2D(first_derivative, -1, Sy) #convolved first derivative with sobel
		
        #rotating first and second derivatives to get desired orientations
		for o in range(orientations):
			fltr_orientation = o * 180 / orientations
     	
			first_derivative =  imutils.rotate(first_derivative, fltr_orientation)
			first_derivatives.append(first_derivative)

			second_derivative = imutils.rotate(second_derivative, fltr_orientation)
			second_derivatives.append(second_derivative)
	
	# 8 such filters are generated
	for scale in LoG_scale:
		sgm = sgm = [scale, scale]
		G = G2d(sgm, fltr_sz)
		laplacian_kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		LoG.append(cv2.filter2D(G, -1, laplacian_kernal)) #convolving laplacian kernel and gaussian2D to get Laplacian of Gaussian filters
		
    # 4 gaussian filters are generated
	for scale in gaussian_scale:
		sgm = [scale, scale]
		gaussian.append(G2d(sgm, fltr_sz))
		
    # total of 18 + 18 + 8 + 4 = 48 filters are generated
	fltr_bank = first_derivatives + second_derivatives + LoG + gaussian
	return fltr_bank


def gaborFilter(scales, orientations, frequencies, fltr_sz):
	fltr_bank = []
	for scale in scales:
		sgm = [scale, scale]
		G = G2d(sgm, fltr_sz)
		for f in frequencies:
			for o in range(orientations):
				fltr_orientation = o * np.pi / orientations
				sine_wave = sin2d(f, fltr_sz, fltr_orientation)
				gabor_fltr = G * sine_wave
				fltr_bank.append(gabor_fltr)

	return fltr_bank


def halfDisk(radius, angle):
	sz = 2*radius + 1
	c = radius
	half_disk = np.zeros([sz, sz])
	for i in range(radius):
		for j in range(sz):
			distance = np.square(i-c) + np.square(j-c)
			if distance <= np.square(radius):
				half_disk[i,j] = 1
    
	
	half_disk = imutils.rotate(half_disk, angle)
	half_disk[half_disk<=0.5] = 0
	half_disk[half_disk>0.5] = 1
	return half_disk

def halfDiskMasks(scales, orients):
	fltr_bank = []
	for r in scales:
		fltr_bank_pairs = []
		temp = []
		for o in range(orients):
			angle = o * 360 / orients
			half_disk_fltr = halfDisk(r, angle)
			temp.append(half_disk_fltr)

		#making pairs
		i = 0
		while i < orients/2:
			fltr_bank_pairs.append(temp[i])
			fltr_bank_pairs.append(temp[i+int((orients)/2)])
			i = i+1

		fltr_bank+=fltr_bank_pairs


	return fltr_bank

def applyFilters(image, fltr):
	out_images = []	
	for ft in fltr:
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		out_image = cv2.filter2D(image_gray,-1, ft)
		out_images.append(out_image)

	return out_images

def chisqDistance(input, bins, fltr):

	chi_sq_distances = []
	N = len(fltr)
	n = 0
	while n < N:
		left_mask = fltr[n]
		right_mask = fltr[n+1]		
		tmp = np.zeros(input.shape)
		chi_sq_dist = np.zeros(input.shape)
		min_bin = np.min(input)
	

		for bin in range(bins):
			tmp[input == bin+min_bin] = 1
			g_i = cv2.filter2D(tmp,-1,left_mask)
			h_i = cv2.filter2D(tmp,-1,right_mask)
			chi_sq_dist += (g_i - h_i)**2/(g_i + h_i + np.exp(-7))

		chi_sq_dist /= 2
		chi_sq_distances.append(chi_sq_dist)
		n = n+2
    	

	return chi_sq_distances

def saveFltrImg(fltr, fl, cols):
	rows = math.ceil(len(fltr)/cols)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(fltr)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(fltr[index], cmap='gray')
	
	plt.savefig(fl)
	plt.close()

def loadImg(fldr, fl):
	print("Loading ", fldr)
	images = []
	if fl == None:
		fl = os.listdir(fldr)
	print(fl)
	for f in fl:
		img_path = fldr + "/" + f
		image = cv2.imread(img_path)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading ", image)

	return images

def jpg2png(a):
	b = []
	for n in range(len(a)):
		s = a[n]
		f_name = str()
		for i in range(len(s)):
			if s[i] == '.':
				break

			f_name += str(s[i])	

		b.append(str(f_name) + ".png")
	
	return b


def pbEdges(T_g, B_g, C_g, edgeCanny, edgesSobel, wgt, i):
	edgeCanny = cv2.cvtColor(edgeCanny, cv2.COLOR_BGR2GRAY)
	edgesSobel = cv2.cvtColor(edgesSobel, cv2.COLOR_BGR2GRAY)
	T1 = (T_g + B_g + C_g)/3
	w1 = wgt[0]
	w2 = wgt[1]
	T2 = (w1 * edgeCanny) + (w2 * edgesSobel)
	pb_lite_op = np.multiply(T1, T2)

	return pb_lite_op

def main():


	get_Tmaps = True
	get_Bmap = True
	get_Cmap = True

	Tbins = 64
	Bbins = 16
	Cbins = 16
 
	Tmaps = []
	Bmaps = []
	C_maps = []

	Tgradients = []
	Bgradients = []
	Cgradients = []
	
	
	folder_name = "./Phase1/"
	img_fldr = folder_name + "BSDS500/Images"
	sobel_fldr = folder_name + "BSDS500/SobelBaseline"
	canny_fldr = folder_name + "BSDS500/CannyBaseline"

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	dog_fltr_bank = DoGFilters(16, [3,5], 50)
	saveFltrImg(dog_fltr_bank, folder_name + "results/Filters/DoG.png", cols = 8)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LMS_fltr_bank = LMFilter([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49) # input = LMsmall filter scales
	saveFltrImg(LMS_fltr_bank,folder_name + "results/Filters/LMS.png", 12)
	LML_fltr_bank = LMFilter([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49) # input = LMlarge filter scales
	saveFltrImg(LML_fltr_bank, folder_name + "results/Filters/LML.png", 12)

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gabor_fltr_bank = gaborFilter([10,30], 12, [3,4,5], 50)
	saveFltrImg(gabor_fltr_bank, folder_name + "results/Filters/Gabor.png",6)


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	halfDisk_fltr_bnk = halfDiskMasks([2,5,10,20,30], 16)
	saveFltrImg(halfDisk_fltr_bnk, folder_name + "results/Filters/HDMasks.png", 8)

	img = loadImg(img_fldr, fl=None)
	fl = os.listdir(img_fldr)
	fltr = dog_fltr_bank + LML_fltr_bank + LMS_fltr_bank + gabor_fltr_bank


	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	if get_Tmaps:
		for i,image in enumerate(img):
			fltr_img = applyFilters(image, fltr)	
			fltr_img = np.array(fltr_img)
			f,x,y = fltr_img.shape
			ip_mat = fltr_img.reshape([f, x*y]) # input matrix
			ip_mat = ip_mat.transpose()

			"""	
			Generate texture ID's using K-means clustering	
			Display texton map and save image as TextonMap_ImageName.png,	
			use command "cv2.imwrite('...)"	
			"""
			kmeans = sklearn.cluster.KMeans(n_clusters = Tbins, n_init = 10)
			kmeans.fit(ip_mat)
			labels = kmeans.predict(ip_mat)
			Timg = labels.reshape([x,y])
			Tmaps.append(Timg)
			plt.imsave(folder_name + "results/Maps/TMap_"+ fl[i], Timg)



	"""	
	Generate Texton Gradient (Tg)	
	Perform Chi-square calculation on Texton Map	
	Display Tg and save image as Tg_ImageName.png,	
	use command "cv2.imwrite(...)"	
	"""
	for i,tmap in enumerate(Tmaps):
		T_g = chisqDistance(tmap, Tbins, halfDisk_fltr_bnk)
		T_g = np.array(T_g)
		T_g = np.mean(T_g, axis = 0)
		Tgradients.append(T_g)
		plt.imsave(folder_name + "results/TGradients/tg_" + fl[i], T_g)	


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	if get_Bmap:
		for i,image in enumerate(img):
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			x,y = image_gray.shape
			ip_mat = image_gray.reshape([x*y,1])
			kmeans = sklearn.cluster.KMeans(n_clusters = Bbins, n_init = 10)
			kmeans.fit(ip_mat)
			labels = kmeans.predict(ip_mat)
			Bimg = labels.reshape([x,y])
			Bmaps.append(Bimg)
			plt.imsave(folder_name + "results/Maps/BMap_" + fl[i], Bimg, cmap = 'gray')


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	for i,bmap in enumerate(Bmaps):
		B_g = chisqDistance(bmap, Bbins, halfDisk_fltr_bnk)
		B_g = np.array(B_g)
		B_g = np.mean(B_g, axis = 0)
		Bgradients.append(B_g)
		plt.imsave(folder_name + "results/BGradients/bg_" + fl[i], B_g, cmap = 'gray')


	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	if get_Cmap:
		for i,image in enumerate(img):
			x,y,c = image.shape
			ip_mat = image.reshape([x*y,c])
		
			kmeans = sklearn.cluster.KMeans(n_clusters = Cbins, n_init = 10)
			kmeans.fit(ip_mat)
			labels = kmeans.predict(ip_mat)
			Cimg = labels.reshape([x,y])
			C_maps.append(Cimg)	
			plt.imsave(folder_name + "results/Maps/CMap_"+ fl[i], Cimg)


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	for i,clr_map in enumerate(C_maps):
		C_g = chisqDistance(clr_map, Cbins, halfDisk_fltr_bnk)
		C_g = np.array(C_g)
		C_g = np.mean(C_g, axis = 0)
		Cgradients.append(C_g)
		plt.imsave(folder_name + "results/CGradients/cg_" + fl[i], C_g)


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	baseline_fls = jpg2png(fl)
	print(baseline_fls)
	sobel= loadImg(sobel_fldr, baseline_fls)


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	canny = loadImg(canny_fldr, baseline_fls)


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	if get_Tmaps and get_Bmap and get_Cmap:
		for i in range(len(img)):	
			print("Edges for image ", baseline_fls[i])	
			pb_edge = pbEdges(Tgradients[i], Bgradients[i], Cgradients[i], canny[i], sobel[i], [0.5,0.5], baseline_fls[i])
			plt.imshow(pb_edge, cmap = "gray")
			plt.show()
			plt.imsave("Phase1/results/final_output/" + baseline_fls[i], pb_edge, cmap = "gray")
    
if __name__ == '__main__':
    main()
 


