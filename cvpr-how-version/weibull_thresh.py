import os
import sys
import numpy as np

def mkdir(path):
        print('mkdir', path)
        try:
                os.mkdir(path)
        except:
                print('warning, cannot make directory')

#-----------------------------
# Read Command Arguments
#-----------------------------
if (len(sys.argv)<4):
        print('Usage:')
        print('')
        print('   python3 weibull_thresh.py dataset model seed')
        print('')
        print('dataset:   cifar10 cifar100 mnist or imagenette2')
        print('model:     resnet18 resnet50 vgg19')
        print('seed:      start with 0 and go from there')
        exit(1)
arg_dataset =       sys.argv[1]
arg_model   =       sys.argv[2]
arg_seed    =   int(sys.argv[3])

print('arg_dataset', arg_dataset)
print('arg_model',   arg_model)
print('arg_seed',    arg_seed)

dist_dir = 'distribution/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
param_dir = dist_dir + '/param'
thresh_dir = dist_dir + '/thresh'

print('dist_dir', dist_dir)
print('param_dir', param_dir)
print('thresh_dir', thresh_dir)

mkdir(thresh_dir)

# Which prefix to use
if arg_model=='vgg19':
	prefixes = ('A', 'B', 'C', 'D', 'E')
else:
	prefixes = ('X', 'A', 'B', 'C', 'D')

#----------------------
# Read the parameter files
#----------------------
for prefix in prefixes:

	#-----------------
	# Read the parameter CSV
	#-----------------
	param_path = param_dir + '/' + prefix + '_weibull.csv'
	print('open', param_path)
	f = open(param_path, 'r')
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

	nFeat = rows-1
	nParam = cols
	print('nFeat', nFeat)
	print('nParam', nParam)
	#input('enter')

	#-------------
	# read lamda and k
	#-------------
	lamda = np.zeros((nFeat,))
	k     = np.zeros((nFeat,))
	for y in range(nFeat):
		lamda[y] = float(csv[y+1][0])
		k[y]     = float(csv[y+1][1])
		print('y', y, 'lamda', lamda[y], 'k', k[y])

	#-------------
	# Calculate thresholds for every 5 percentile
	#-------------
	perc_step = 5
	for percentile in range(perc_step, 100, perc_step):
		print('percentile', percentile)


		#-------------
		# Closed form inverse cdf formula
		#-------------
		alpha  = 0.01 * percentile
		rhs    = -np.log(1.0 - alpha)
		thresh = lamda * np.power(rhs, 1.0/k)

#		print('thresh', thresh)
#		input('enter')

		# Write out the thresholds
		thresh_path = thresh_dir + '/%s_weibull_%02d.csv' % (prefix, percentile)
		print('thresh_path', thresh_path)
		fthresh = open(thresh_path, 'w')
		for j in range(nFeat):
			fthresh.write('%f\t\n' % thresh[j])
		fthresh.close()


	'''
	#-----------------
	# copy to a table to numpy
	#-----------------
	data = np.zeros((nFeat,nField))
	for y in range(nFeat):
		for x in range(nField):
			data[y,x] = float(csv[y+1][x+1])

	print('data', data)
	print('data', data.shape)
	#input('enter')

	#-----------------
	# extract field names
	#-----------------
	fields = [None] * nField
	for y in range(nField):
		fields[y] = csv[0][y+1]

	#-----------------
	# Calculate mean and stdev
	#-----------------
	mean = np.mean(data,axis=0)
	#print('mean', mean)
	#print('mean', mean.shape)
	#input('enter')

	stdev = np.mean(data,axis=0)
	#print('stdev', stdev)
	#print('stdev', stdev.shape)
	#input('enter')

	mini = np.min(data,axis=0)
	#print('mini', mini)
	#print('mini', mini.shape)
	#input('enter')

	maxi = np.max(data,axis=0)
	#print('maxi', maxi)
	#print('maxi', maxi.shape)
	#input('enter')

	stderr = stdev / np.sqrt(nFeat)
	#print('stderr', stderr)
	#print('stderr', stderr.shape)
	#input('enter')

	lo = mean - stderr
	#print('lo', lo)
	#print('lo', lo.shape)
	#input('enter')

	hi = mean + stderr
	#print('hi', hi)
	#print('hi', hi.shape)
	#input('enter')

	#-----------------------------------
	# Write the output table
	#-----------------------------------
	write_table_fields(fields)
	write_table_entry(prefix, 'mean', mean)
	write_table_entry(prefix, 'lo', lo)
	write_table_entry(prefix, 'hi', hi)
	write_table_entry(prefix, 'stderr', stderr)
	write_table_entry(prefix, 'stdev', stdev)
	write_table_entry(prefix, 'mini',mini)
	write_table_entry(prefix, 'maxi',maxi)
	'''

