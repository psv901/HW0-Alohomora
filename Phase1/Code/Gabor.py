import numpy as np
import matplotlib.pyplot as plt
import math

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

def sin2d(frequency, sz, theta):
	if (sz%2) == 0:
		idx = sz/2
	else:
		idx = (sz - 1)/2

	x, y = np.meshgrid(np.linspace(-idx, idx+1, sz), np.linspace(-idx, idx+1, sz))
	z = x * np.cos(theta) + y * np.sin(theta)
	sin2d = np.sin(z * 2 * np.pi * frequency/sz)

	return sin2d

def gaborFilter(scales, orientations, frequencies, fltr_size):
	fltr_bank = []
	for scale in scales:
		sgm = [scale, scale]
		G = G2d(sgm, fltr_size)
		for f in frequencies:
			for o in range(orientations):
				fltr_orientation = o * np.pi / orientations
				sine_wave = sin2d(f, fltr_size, fltr_orientation)
				gabor_fltr = G * sine_wave
				fltr_bank.append(gabor_fltr)

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
	folder_name = "./Phase1/"
	
	gabor_fltr_bank = gaborFilter([10,30], 12, [3,4,5], 50)
	saveFltrImg(gabor_fltr_bank, folder_name + "results/Gabor.png",6)
	
if __name__ == '__main__':
    main()
