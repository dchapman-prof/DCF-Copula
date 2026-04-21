
import os
import sys


def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print('cannot make '+path)


def system(cmd):
	print('----------------------------------------------------')
	print(' histo.py RUN COMMAND')
	print('', cmd)
	print('----------------------------------------------------')
	rt = os.system(cmd)
	if (rt/8==0):
		print('----------------------------------------------------')
		print(' COMMAND SUCCESSFUL')
		print('----------------------------------------------------')
	else:
		print('----------------------------------------------------')
		print(' COMMAND FAILURE')
		print('----------------------------------------------------')


#----------------------------
# Check command arguments
#----------------------------
if (len(sys.argv)<8):
	print('Usagye')
	print('')
	print('   python3 histo.py dataset model seed minx maxx nbin maxC1')
	print('')
	print('  dataset:   cifar10 cifar100 imagenette2 mnist')
	print('  model:     resnet18 resnet50 vgg19')
	print('  seed:      integer seed for which model to load')
	print('  batchsize: how many images to process at once')
	print('  minx:      minimum cutoff for histogram')
	print('  maxx:      maximum cutoff for histogram')
	print('  nbin:      number of bins for histogram')
	print('  maxC1:     maximum number of channels to create histograms')
	sys.exit(1)

arg_dataset    =         sys.argv[1]
arg_model      =         sys.argv[2]
arg_seed       =     int(sys.argv[3])
arg_minx       =   float(sys.argv[4])
arg_maxx       =   float(sys.argv[5])
arg_nbin       =     int(sys.argv[6])
arg_maxC1      =     int(sys.argv[7])

print('arg_dataset', arg_dataset)
print('arg_model', arg_model)
print('arg_seed', arg_seed)
print('arg_minx', arg_minx)
print('arg_maxx', arg_maxx)
print('arg_nbin',  arg_nbin)
print('arg_maxC1', arg_maxC1)

#
# Construct the folders
#
featdir = 'features/%s_%s_%d' % (arg_dataset,arg_model,arg_seed)
histdir = 'histogram_2d/%s_%s_%d' % (arg_dataset,arg_model,arg_seed)
mkdir('histogram_2d')
mkdir(histdir)


#
# Compile the C program
#
system("make")

#
# For every feature array
#
for arr in ('X_train', 'A_train', 'B_train', 'C_train', 'D_train', 'E_train', 'X_test', 'A_test', 'B_test', 'C_test', 'D_test', 'E_test'):

	#cmd = './histo_kernel %s/%s.bin %s/%s.csv %f %f %d %f' % (featdir,arr,histdir,arr,arg_minx,arg_maxx,arg_nbin,arg_width)
	cmd = './histo_2d %s/%s.bin %s/%s.bin %f %f %d %d' % (featdir,arr,histdir,arr,arg_minx,arg_maxx,arg_nbin,arg_maxC1)
	system(cmd)


print('Done!\n')

