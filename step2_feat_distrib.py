import os
import sys

def system(cmd):
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print(' RUN COMMAND')
	print('', cmd)
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	rt = os.system(cmd)
	if (rt/8==0):
		print('!!!!!!!!!!!!!!!!!!!!!!')
		print('! COMMAND SUCCESSFUL !')
		print('!!!!!!!!!!!!!!!!!!!!!!')
	else:
		print('!!!!!!!!!!!!!!!!!!!')
		print('! COMMAND FAILURE !')
		print('!!!!!!!!!!!!!!!!!!!')


print('##########################################################################')
print('##########################################################################')
print('##########################################################################')
print('###                 BEGIN STEP 2 FEATURE DISTRIBUTION                  ###')
print('##########################################################################')
print('##########################################################################')
print('##########################################################################')

#
# Compile the C program
#
system('make')

#---------------------------
# for all seeds, models, and datasets
#---------------------------
for seed in range(5):
	for model in ('resnet50', 'vgg19', 'resnet18'):
		for dataset in ('mnist', 'cifar10', 'cifar100', 'imagenette2'):
		#for dataset in ('imagenette2',):

			#
			# Extract the features
			#
			#system('python3 extract_feat.py %s %s %d 256' % (dataset, model, seed))

			#
			# Extract the histogram
			#
			#system('python3 histo.py %s %s %d -10 10 2000 0.05' % (dataset, model, seed))

			#
			# Fit the feature distribution
			#
			system('./fit_distr %s %s %d' % (dataset, model, seed))

			#
			# Make the tables
			#
			system('python3 table_distr.py %s %s %d' % (dataset, model, seed))

			#
			# Plot the feature distribution
			#
			system('python3 plot_distr.py %s %s %d -0.5 2.0' % (dataset, model, seed))
			system('python3 plot_distr_clean.py %s %s %d -0.5 2.0' % (dataset, model, seed))

			#
			# Extract all of the weibull thresholds
			#
			system('python3 weibull_thresh.py %s %s %d' % (dataset, model, seed))




print('##########################################################################')
print('##########################################################################')
print('##########################################################################')
print('###                  END STEP 2 FEATURE DISTRIBUTION                   ###')
print('##########################################################################')
print('##########################################################################')
print('##########################################################################')
