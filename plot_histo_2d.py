import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readBin(fname):
	#-----
	# Read the metadata
	#-----
	print('read',fname+'.txt')
	fin = open(fname+'.txt', 'r')
	str = fin.read().split()
	fin.close()
	K = len(str)-1
	dims=[0]*K
	N = 1
	for k in range(K):
		dims[k] = int(str[k])
		N *= dims[k]
	
	if (str[K]=='int32'):
		print('int32')
		dtype = np.int32
	else:
		print('float32')
		dtype = np.float32
	
	print('dims', dims)
	print('N', N)
	
	#-----
	# Read the binary data
	#-----
	print('read',fname)
	array = np.fromfile(fname, dtype=np.int32, count=N)
	array = np.reshape(array, dims)
	
	return array


def mkdir(path):
	print('mkdir', path)
	try:
		os.mkdir(path)
	except:
		print('warning, cannot make directory')

#------------------
# Read Command Arguments
#------------------
if (len(sys.argv)<6):
	print('Usage:')
	print('')
	print('   python3 plot_histo_2d.py dataset model seed minx maxx')
	print('')
	print('dataset:   cifar10 cifar100 mnist or imagenette2')
	print('model:     resnet18 resnet50 vgg19')
	print('seed:      start with 0 and go from there')
	exit(1)
arg_dataset =       sys.argv[1]
arg_model   =       sys.argv[2]
arg_seed    =   int(sys.argv[3])
arg_minx    = float(sys.argv[4])
arg_maxx    = float(sys.argv[5])

print('arg_dataset', arg_dataset)
print('arg_model',   arg_model)
print('arg_seed',    arg_seed)

plot_folder = 'histogram_2d/%s_%s_%d/plot' % (arg_dataset, arg_model, arg_seed)
mkdir(plot_folder)

#------------------
# Make all the plots
#------------------
for arr in ('X_train', 'A_train', 'B_train', 'C_train', 'D_train', 'E_train', 'X_test', 'A_test', 'B_test', 'C_test', 'D_test', 'E_test'):

	hfile = 'histogram_2d/%s_%s_%d/%s.bin' % (arg_dataset, arg_model, arg_seed, arr)
	histo = readBin(hfile)
	
	maxC1 = histo.shape[0]
	maxC2 = histo.shape[1]
	
	# Example grid of values (2D NumPy array)
	grid = np.random.rand(10, 10)  # Replace with your actual data

	for c1 in range(maxC1):
		for c2 in range(maxC2):

			grid = np.log10(histo[c1][c2]+0.1)

			# Define the x and y bounds
			x_min, x_max = arg_minx, arg_maxx
			y_min, y_max = arg_minx, arg_maxx

			plt.figure(1, figsize=(10,10))

			# Create the heatmap
			plt.imshow(grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', aspect='auto')

			# Add a color bar
			plt.colorbar()

			# Add labels
			plt.xlabel('X-axis')
			plt.ylabel('Y-axis')

			# Save the figure
			plot_path = '%s/%s_%d_%d.png' % (plot_folder, arr, c1, c2)
			print('save', plot_path)
			plt.savefig(plot_path)
			
			# clear the figure
			plt.clf()


print('Success!')
	
	
		
	
	
