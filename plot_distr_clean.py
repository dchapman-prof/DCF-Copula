import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fontsize=16

matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize) 


def mkdir(path):
	print('mkdir', path)
	try:
		os.mkdir(path)
	except:
		print('warning, cannot make directory')

def parserow(csv_row):
	nX = len(csv_row) - 3
	rowdata = np.zeros((nX,))
	for x in range(nX):
		rowdata[x] = float(csv_row[x+2])
		#print('rowdata[%d] %f' % (x,rowdata[x]))
		#input('enter')
	return rowdata

def parseargv(idx):
	if len(sys.argv)<=idx:
		return None
	if sys.argv[idx]=='nan':
		return None
	val = float(sys.argv[idx])
	if val==-9999.0:
		return None
	return val

#-----------------------------
# Read Command Arguments
#-----------------------------
if (len(sys.argv)<5):
	print('Usage:')
	print('')
	print('   python3 plot_distr.py dataset model seed [x0 x1]')
	print('')
	print('dataset:   cifar10 cifar100 mnist or imagenette2')
	print('model:     resnet18 resnet50 vgg19')
	print('seed:      start with 0 and go from there')
	print('x0:        minval   -9999.0 or nan or missing means autoplot')
	print('x1:        maxval   -9999.0 or nan or missing means autoplot')
	exit(1)
arg_dataset =       sys.argv[1]
arg_model   =       sys.argv[2]
arg_seed    =   int(sys.argv[3])
arg_x0      =      parseargv(4)
arg_x1      =      parseargv(5)
arg_xlim = (arg_x0!=None and arg_x1!=None)

print('arg_dataset', arg_dataset)
print('arg_model',   arg_model)
print('arg_seed',    arg_seed)
print('arg_x0',      arg_x0)
print('arg_x1',      arg_x1)
print('arg_xlim',    arg_xlim)


#---------------------
# Parse directories
#---------------------

dist_dir = 'distribution/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
plot_dir = dist_dir + '/plot_clean_%f_%f' % (arg_x0, arg_x1)
print('dist_dir', dist_dir)
print('plot_dir', plot_dir)

mkdir(plot_dir)


#----------------------
# Read the histogram files
#----------------------
for prefix in ('X', 'A', 'B', 'C', 'D', 'E'):

	#-----------------
	# Read the histogram CSV
	#-----------------
	histo_path = dist_dir + '/' + prefix + '_histogram.csv'
	print('open', histo_path)
	try:
		f = open(histo_path, 'r')
	except:
		print('cannot open '+histo_path+' skipping')
		continue
	csv = list(f.readlines())
	f.close()
	rows = len(csv)
	cols = 0
	for i in range(rows):
		csv[i] = csv[i].split('\t')
		cols = max(cols, len(csv[i]) - 1)   # minus 1 to remove trailing tab

	print('rows', rows)
	print('cols', cols)
	#input('enter')


	#------------------
	# Read the ticks
	#------------------
	nX      = cols-2
	nCurves = rows-1

	ticks = parserow(csv[0])
	#input('enter')

	#------------------
	# For every feature (until all curves are read)
	#------------------
	figno = 0
	curveno = 0
	while (curveno < nCurves):

		print('-------------------------------------')
		print(' %s figno %d' % (prefix, figno))
		print('-------------------------------------')

		# Extract the feature number
		feature = int(csv[curveno+1][0])
		print('feature', feature)

		# Extract all curves for this feature
		curves      = []
		distr_names = []
		while True:
			distr_names.append( csv[curveno+1][1] )   # append the name of the distribution
			curves.append( parserow(csv[curveno+1]) ) # append the numeric data
			curveno+=1                                # we've read the curve
			if curveno>=nCurves:                      # are we done with all curves ?
				break
			next_feature = int(csv[curveno+1][0])     # is the next curve part of the same feature?
			if (next_feature != feature):
				break

		# Which distributions did we read ?
		#print('distr_names', distr_names)
		#input('enter')

		#for i in range(len(distr_names)):
		#	print(distr_names[i], 'min', np.min(curves[i]), 'max', np.max(curves[i]))
		#input('enter')

		#for y in range(curves[0].shape[0]):
		#	print('ticks', ticks[y], end=' ')
		#	for i in range(len(curves)):
		#		print(distr_names[i], curves[i][y], end=' ')
		#	print('')
		#	input('enter')

		#
		# HACK the curves to only plot train and test
		#
		curves = curves[0:2]

		#---------------------------
		# make the plot
		#---------------------------
		plt.figure(1, figsize=(10,10))
		fig,ax = plt.subplots()
		for i in range(len(curves)):
			curve      = curves[i]
			distr_name = distr_names[i]
			ax.plot(ticks, curve, '-', label=distr_name)
		leg = ax.legend(loc='upper right', frameon=False, fontsize=str(fontsize))
		if (arg_xlim):
			ax.set_xlim(arg_x0, arg_x1)
		plot_path = '%s/%s_%05d.png' % (plot_dir, prefix, feature)
		print('save', plot_path)
		plt.savefig(plot_path)

		# On to the next figure
		figno+=1


print('Done!')
