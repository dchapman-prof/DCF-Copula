import os
import sys
import numpy as np

#-----------------------------
# Read Command Arguments
#-----------------------------
if (len(sys.argv)<4):
        print('Usage:')
        print('')
        print('   python3 table_distr.py dataset model seed')
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

table_path = dist_dir + '/table.csv'
ftable = open(table_path, "w")

def write_table_entry(prefix,name,vals):
	ftable.write('%s\t%s\t' % (prefix, name))
	for x in range(vals.shape[0]):
		ftable.write('%.6f\t' % vals[x])
	ftable.write('\n')
	ftable.flush()

def write_table_fields(fields):
	ftable.write('\t\t')
	for x in range(len(fields)):
		ftable.write('%s\t' % fields[x])
	ftable.write('\n')
	ftable.flush()


#----------------------
# Read the loss files
#----------------------
for prefix in ('X', 'A', 'B', 'C', 'D', 'E'):

	#-----------------
	# Read the loss CSV
	#-----------------
	loss_path = dist_dir + '/' + prefix + '_loss.csv'
	print('open', loss_path)
	try:
		f = open(loss_path, 'r')
		csv = list(f.readlines())
		f.close()
	except:
		print('WARNING: cannot open', loss_path, 'for reading')
		continue
	rows = len(csv)
	cols = 0
	for i in range(rows):
		csv[i] = csv[i].split('\t')
		cols = max(cols, len(csv[i]) - 1)   # minus 1 to remove trailing tab


	print('rows', rows)
	print('cols', cols)
	#input('enter')

	nFeat = rows-1
	nField = cols-1
	print('nFeat', nFeat)
	print('nField', nField)
	#input('enter')

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


ftable.close()

print('Done!\n')
