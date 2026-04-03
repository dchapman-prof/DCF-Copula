import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def stripvec(csv, lineno):
	v = np.zeros((5,), dtype=np.float32)
	for i in range(5):
		v[i] = float(csv[lineno][i+2])
	return v

def PlotTable(dataset, model):
	prefix = '%s_%s' % (dataset, model)
	incsv = prefix + '.csv'
	outpng = incsv + '.png'
	print('prefix', prefix)
	print('incsv', incsv)
	print('outpng', outpng)
	
	#
	# Read the CSV file
	#
	f = open(incsv, 'r')
	csv = list(f.readlines())
	for i in range(len(csv)):
		csv[i] = csv[i].split('\t')
	f.close()
	
	#print('csv')
	#print(csv)
	
	#
	# Parse into the table
	#
	mean = np.zeros((5,5), dtype=np.float32)
	lo   = np.zeros((5,5), dtype=np.float32)
	hi   = np.zeros((5,5), dtype=np.float32)
	
	lineno = 1
	for i in range(5):
		mean[i]=stripvec(csv, lineno)
		lo[i]=stripvec(csv, lineno+1)
		hi[i]=stripvec(csv, lineno+2)
		lineno+=8
	
	print('mean', mean)
	print('lo', lo)
	print('hi', hi)

	#
	# Make the plot
	#

	fig, ax = plt.subplots(figsize=(8,6))
	
	# Create a figure and axes
	x = list(range(5))
	
	colors = ('red', 'orange', 'green', 'blue', 'violet')
	series = ('uniform', 'gaussian', 'exponential', 'gamma', 'weibull')
	for i in range(5):
		print('------------------')
		print('i', i)
		print('x', x)
		print('lo[i]', lo[i])
		print('hi[i]', hi[i])
		print('mean[i]', mean[i])
		print('colors[i]', colors[i])
		print('series[i]', series[i])
		ax.fill_between(x, lo[:,i], hi[:,i], color=colors[i], alpha=0.3)
		ax.plot(x, mean[:,i], color=colors[i], label=series[i])

	# Add labels and title
	#ax.tick_params(axis='both', which='major', pad=15)
	plt.xlabel('Feature Layer', labelpad=15)
	plt.ylabel('KL-divergence', labelpad=15)
	plt.xticks((0,1,2,3,4))
	plt.title('%s %s' % (dataset, model))
	plt.legend(loc='upper right')

	#plt.show()
	plt.savefig(outpng)


for dataset in ['cifar10', 'cifar100', 'imagenette2', 'mnist']:
	for model in ['resnet18', 'resnet50', 'vgg19']:
		PlotTable(dataset, model)
		#input('press enter')

s