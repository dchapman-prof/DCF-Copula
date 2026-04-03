
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
if (len(sys.argv)<6):
	print('Usagye')
	print('')
	print('   python3 quantile.py dataset.py model.py seed nbin nquant')
	print('')
	print('  dataset:   cifar10 cifar100 imagenette2 mnist')
	print('  model:     resnet18 resnet50 vgg19')
	print('  seed:      integer seed for which model to load')
	print('  nbin:      number of bins for histogram')
	print('  nquant:    number of quantiles')
	sys.exit(1)

arg_dataset    =         sys.argv[1]
arg_model      =         sys.argv[2]
arg_seed       =     int(sys.argv[3])
arg_nbin       =     int(sys.argv[4])
arg_nquant     =     int(sys.argv[5])

print('arg_dataset', arg_dataset)
print('arg_model', arg_model)
print('arg_seed', arg_seed)
print('arg_nbin',   arg_nbin)
print('arg_nquant',   arg_nquant)

#
# Construct the folders
#
featdir = 'features/%s_%s_%d' % (arg_dataset,arg_model,arg_seed)
quantdir = 'quantiles/%s_%s_%d_%d' % (arg_dataset,arg_model,arg_seed,arg_nquant)
mkdir('quantiles')
mkdir(quantdir)


#
# Compile the C program
#
system("make")

#
# For every feature array
#
for arr in ('X_train', 'A_train', 'B_train', 'C_train', 'D_train', 'E_train', 'X_test', 'A_test', 'B_test', 'C_test', 'D_test', 'E_test'):

	cmd = './quantile %s/%s.bin %s/%s.csv %d %d' % (featdir,arr,quantdir,arr,arg_nbin,arg_nquant)
	system(cmd)


print('Done!\n')

