import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize_histogram(hist_matrix, num_bins=100):

	#num_bins = 10
	deltax   = deltay = 2 / num_bins
	#print('num_bins', num_bins)
	#input("enter")
	#print('deltax', deltax)
	#input("enter")
	

	# Compute the double integral J = sum(h(x,y) * deltax * deltay)
	##norm = np.sum(hist_matrix * deltax * deltay)
	
	total_sum = np.sum(hist_matrix)
	#print('total_sum', total_sum)
	#input("enter")
	
	norm  = total_sum * deltax * deltay
	#print('norm', norm)
	#input("enter")
	
		
	# Normalize the histogram by dividing each element by J
	normalized_histogram = hist_matrix / norm
	    

	return normalized_histogram, norm


def plot_histogram(csv_file,plot_folder):
	fontsize =16
	
	#--------------------
	# Load the CSV
	#--------------------
	hist_data = pd.read_csv(csv_file)

	#--------------------
	# BinA and BinB labels
	#--------------------
	bin_a_labels = hist_data['BinA/BinB'].values
	bin_b_labels = hist_data.columns[1:].astype(float)

	#--------------------
	# Counts from CSV data
	#--------------------
	hist_matrix = hist_data.iloc[:, 1:].values
	#print("hist_matrix",hist_matrix)
	#input("enter")
	
	#--------------------
	# Normalize the data 
	#--------------------
	#if normalize:
		#hist_matrix = hist_matrix / np.max(hist_matrix) 
	#num_bins =10
	#normalized_histogram, norm = normalize_histogram(hist_matrix, num_bins)
	#print("normalized_histogram",normalized_histogram)
	#input("enter")
	
	#print(f"Normalization factor (norm) for {csv_file}: {norm}") 

	#--------------------
	# Create plot
	#--------------------

	bin_a_mesh, bin_b_mesh = np.meshgrid(np.arange(len(bin_b_labels)), np.arange(len(bin_a_labels)))

	#--------------------
	# Plot the 2D histogram as a heatmap
	#--------------------

	plt.figure(figsize=(10, 6))
	plt.pcolormesh(bin_b_mesh, bin_a_mesh, hist_matrix, shading='auto', cmap='viridis', vmin = 0 , vmax = 0.5)
	#plt.pcolormesh(bin_b_mesh, bin_a_mesh, normalized_histogram, shading='auto', cmap='viridis', vmin = 0 , vmax =0.5)

	#--------------------
	# Set axis labels and title
	#--------------------
	#plt.xlabel('BinB', fontsize = fontsize)
	#plt.ylabel('BinA', fontsize = fontsize)
	#plt.title('2D Histogram', fontsize = fontsize)

	#--------------------
	# Tickness of BinA and BinB based on labels
	#--------------------
	nbins = len(bin_b_labels)-1
	bin_pos   = [0.0*nbins, 0.25*nbins, 0.5*nbins, 0.75*nbins, 1.0*nbins]
	bin_label = [-1.0, -0.5, 0.0, 0.5, 1.0] 
	#plt.xticks(np.arange(len(bin_b_labels)) + 0.5, bin_b_labels)
	#plt.yticks(np.arange(len(bin_a_labels)) + 0.5, bin_a_labels)
	plt.xticks(bin_pos, bin_label, fontsize = fontsize)
	plt.yticks(bin_pos, bin_label, fontsize = fontsize)

	#--------------------
	# Add color bar to represent the counts
	#--------------------
	cbar = plt.colorbar(label='Count')
	cbar.set_label('Count', fontsize = fontsize)
	cbar.ax.tick_params(labelsize=fontsize)
	
	#--------------------
	# Save the plot to a file
	#--------------------
	plot_filename = os.path.join(plot_folder, os.path.basename(csv_file).replace('.csv', '.png'))
	plt.savefig(plot_filename)
	plt.close()  

	print(f"Plot saved as {plot_filename}")


    	

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
print('###                     BEGIN STEP2_plot_histogram                     ###')
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
# Read Base Directories
#------------------

#base_input_folder = '/home/developers/small_feat/moments/outputs_latest_version/'
base_input_folder = '/home/developers/small_feat/moments/test'
num_bins =100
#-----------------------
# Compile the C program
#-----------------------
system('make')

#----------------------
# Make all the Plots
#----------------------
for dataset_model_seed_folder in os.listdir(base_input_folder):
	full_folder_path = os.path.join(base_input_folder, dataset_model_seed_folder)

	if os.path.isdir(full_folder_path):  
		print(f"Processing folder: {full_folder_path}")

		for file in os.listdir(full_folder_path):
			
			if file.endswith('_pdf.csv'):
			
				if 'Legendre_pdf' in file:
					plot_folder = os.path.join(full_folder_path, 'Legendre_pdf')
				elif 'Fourier_pdf' in file:
                    			plot_folder = os.path.join(full_folder_path, 'Fourier_pdf')
				
				#plot_folder = os.path.join(full_folder_path, 'pdf_plots')
				os.makedirs(plot_folder, exist_ok=True)
				
				full_pdf_path = os.path.join(full_folder_path, file)
                		#print(f"Processing file: {full_csv_path}")

				try:
					plot_histogram(full_pdf_path, plot_folder)
				
				except KeyError as e:
					print(f"Error processing {full_pdf_path}: {e}")
					
					
			elif file.endswith('_hist.csv'):
				
				plot_folder_hist = os.path.join(full_folder_path, 'hist_plots')
				os.makedirs(plot_folder_hist, exist_ok=True)
				
				full_hist_path = os.path.join(full_folder_path, file)
                		#print(f"Processing file: {full_hist_path}")

				try:
					plot_histogram(full_hist_path, plot_folder_hist)
				
				except KeyError as e:
					print(f"Error processing {full_hist_path}: {e}")
            
