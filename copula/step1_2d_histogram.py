import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from hist import create_2d_hist, save_hist


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
print('###                 BEGIN STEP 1_2d_Histogram of pit                   ###')
print('##########################################################################')
print('##########################################################################')
print('##########################################################################')

def mkdir(path):
	print('mkdir', path)
	try:
		os.mkdir(path)
	except:
		print('warning, cannot make directory')

#------------------
# Read Command Arguments
#------------------
arg_minx     = 0
arg_maxx     = 10
arg_nbin     = 100
#arg_nbin     = 11
#------------------
# Read Base Directories
#------------------

base_input_folder = '/home/developers/small_feat/features/'
#base_input_folder = '/home/developers/small_feat/attacked_features/'

base_output_folder = '/home/developers/small_feat/moments/test'
mkdir(base_output_folder)

#-----------------------
# Compile the C program
#-----------------------
system('make')

#----------------------
# Make all the Histograms
#----------------------

train_test_map = {
	'A_train': 'A_test',
	'B_train': 'B_test',
	'C_train': 'C_test',
	'D_train': 'D_test',
	'E_train': 'E_test'
}


for seed in range(1):
	
	#print('10 seed', seed)
	
	#for model in ('resnet18', 'resnet50','vgg19'):
	for model in ('resnet18',):
	#for model in ('vgg19',):
		
		#print('20 model', model)
		
		#for dataset in ('imagenette2', 'mnist', 'cifar10', 'cifar100'):
		#for dataset in ('mnist', 'cifar10', 'cifar100'):
		for dataset in ('imagenette2',):
			
			#print('30 dataset', dataset)
			
			if model == 'vgg19':
			
                		#arr_list = ('A_train', 'B_train', 'C_train', 'D_train', 'E_train', 'A_test', 'B_test', 'C_test', 'D_test', 'E_test')
                		#arr_list = ('A_train','B_train','C_train', 'D_train', 'E_train')
                		arr_list = ('A_train',)
			else:
                		#arr_list = ('A_train', 'B_train', 'C_train', 'D_train', 'A_test', 'B_test', 'C_test', 'D_test')
                		arr_list = ('A_train','B_train','C_train', 'D_train')
                		#arr_list = ('D_train',)

			#print('40 arr_list', arr_list)

			
			for arr in arr_list:
				
				#print('50 arr', arr)
				
				input_folder = os.path.join(base_input_folder, '%s_%s_%d' % (dataset, model, seed))
				output_folder = os.path.join(base_output_folder, '%s_%s_%d' % (dataset, model, seed))
				mkdir(output_folder)
				
				#print('60 input_folder', input_folder)
				#print('70 input_folder', output_folder)
				
				infile1 = os.path.join(input_folder, '%s.bin' % arr)
				
				test_file = train_test_map.get(arr, None)
				
				if test_file:
				
					infile2 = os.path.join(input_folder, '%s.bin' % test_file)
				  
                
				
				#infile2 = os.path.join(input_folder, '%s.bin' % arr)

				#print('80 infile1', infile1)
				#print('90 infile2', infile2)

				for _ in range(1):
				
					print('100 _', _)
				
					#arg_featA = random.randint(0, 10)  
					#arg_featB = random.randint(0, 10)
					#arg_featA_test = random.randint(0, 10)
					#arg_featB_test = random.randint(0, 10)
					
					arg_featA      = 4
					arg_featB      = 2
					arg_featA_test = 4
					arg_featB_test = 2
				

					print('110 arg_featA', arg_featA)
					print('120 arg_featB', arg_featB)
					print('130 arg_featA_test', arg_featA_test)
					print('140 arg_featB_test', arg_featB_test)


					#outfile1 = os.path.join(output_folder, '%s_%s_%d_%s_%d_%d_hist.csv' %(dataset, model, seed, arr, arg_featA, arg_featB)) 
					#outfile2 = os.path.join(output_folder, '%s_%s_%d_%s_%d_%d_pdf.csv' %(dataset, model, seed, arr, arg_featA, arg_featB))
					outfile1 = os.path.join(output_folder, '%s_%d_%d_%s_%d_%d_hist.csv' %(arr, arg_featA, arg_featB, test_file, arg_featA_test,arg_featB_test))
					outfile2 = os.path.join(output_folder, '%s_%d_%d_%s_%d_%d_Legendre_pdf.csv' %(arr, arg_featA, arg_featB, test_file, arg_featA_test, arg_featB_test))
					outfile3 = os.path.join(output_folder, '%s_%d_%d_%s_%d_%d_Fourier_pdf.csv' %(arr, arg_featA, arg_featB, test_file, arg_featA_test, arg_featB_test))
					outfile4 = os.path.join(output_folder, '%s_%d_%d_Legendre_moments.csv' %(arr, arg_featA, arg_featB))
					outfile5 = os.path.join(output_folder, '%s_%d_%d_Fourier_moments.csv' %(arr, arg_featA, arg_featB))
					outfile6 = os.path.join(output_folder, '%s_%s_%s_%d_loss.csv' %(dataset, model, arr, arg_nbin))
					
					#outfile1 = os.path.join(output_folder, '%s_%d_%d_hist.csv' %(arr, arg_featA, arg_featB))
					#outfile2 = os.path.join(output_folder, '%s_%d_%d_pdf.csv' %(arr, arg_featA, arg_featB))
					#----------------------------
					# run characteristic function on Copula
					#-----------------------------
			
					system(f'./pit {infile1} {infile2} {outfile1} {outfile2} {outfile3} {outfile4} {outfile5} {outfile6} {arg_featA} {arg_featB} {arg_featA_test} {arg_featB_test} {arg_minx} {arg_maxx} {arg_nbin}')
					input("Enter")

					#if outfile1.endswith('_hist.csv'):
				        # Load the CSV file
						#data = pd.read_csv(outfile1)
						# Extract BinA, BinB, and Count
						#bin_a = data['BinA'].values
						#bin_b = data['BinB'].values
						#counts = data['Count'].values

						# Create the 2D histogram
						#u_a, u_b, hist_2d = create_2d_hist(bin_a, bin_b, counts)

						# Save the 2D histogram to a new CSV file
						#output_file = os.path.join(output_folder, f'2d_{os.path.basename(outfile1)}')
						#save_hist(output_file, u_a, u_b, hist_2d)

						#print(f"2D histogram saved to {output_file}")

						#input("Enter")
						#print("Success")





			
			
