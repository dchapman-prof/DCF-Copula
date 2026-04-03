import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_histogram(csv_file):
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
		#hist_matrix = 0.5 * hist_matrix / np.max(hist_matrix) 
	hist_matrix *= 0.25
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
	plot_filename = csv_file.replace('.csv', '.png')
	plt.savefig(plot_filename)
	plt.close()  

	print(f"Plot saved as {plot_filename}")

#----------------
# Read command arguments
#----------------
if (len(sys.argv)!=2):
	print("  usage:")
	print("python3 pit_plot.py csv_file")
	sys.exit(1)
	
csv_file = sys.argv[1]
print("csv_file", csv_file)

plot_histogram(csv_file)

print("Done!")



