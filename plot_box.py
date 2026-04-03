import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

if len(sys.argv) != 2:
	print("Usage: python boxplot_theta.py <dataset_model_seed_nquant_percentile_variant>")
	sys.exit(1)

#---------------------
# Read the argument 
#---------------------
dataset_model_seed_percentile_variant = sys.argv[1]

#---------------------
# Split the input argument into components
#---------------------
components = dataset_model_seed_percentile_variant.split('_')


if len(components) != 6:
	print("Error: Argument format is incorrect. Expected format: dataset_model_seed_nquant_percentile_variant")
	sys.exit(1)

#---------------------
# Extract values
#---------------------

dataset      = components[0]
model        = components[1]
seed         = components[2]
nquant       = components[3]
percentile   = components[4]
variant_flag = components[5]

#-----------------------------
# Define the directory path based on dataset_model_seed
#-----------------------------

base_directory = "/home/developers/small_feat/qdistribution/"
plot_directory = os.path.join(base_directory, "plot")
os.makedirs(plot_directory, exist_ok=True)

csv_directory = os.path.join(base_directory, f"{dataset}_{model}_{seed}_{nquant}_{percentile}_{variant_flag}/param/")

box_plot_directory = os.path.join(plot_directory, "box_plot")
os.makedirs(box_plot_directory, exist_ok=True)

if not os.path.exists(csv_directory):
	print(f"Error: Directory '{csv_directory}' does not exist.")
	sys.exit(1)

#-----------------------------
# Extracting Model name
#-----------------------------
if model in ['resnet18', 'resnet50']:
	layer_names = ['X', 'A', 'B', 'C', 'D']
elif model == 'vgg19':
	layer_names = ['A', 'B', 'C', 'D', 'E']
else:
	print(f"Error: Unsupported model type '{model}'")
	sys.exit(1)

#-----------------------------
# List all CSV files in the directory
#-----------------------------
csv_files = os.listdir(csv_directory)

#-----------------------------
# Initialize a dictionary to store theta values per layer
#-----------------------------

theta_values = {}



for i, layer_name in enumerate(layer_names):
   
	layer_csv_file = f"{layer_name}_weibull.csv"
    

	if layer_csv_file in csv_files:
		file_path = os.path.join(csv_directory, layer_csv_file)
		
		#--------------
		# Read the CSV file
		#--------------
		df = pd.read_csv(file_path, header=None, delimiter='\t')
		#print(df)
		#input("enter")
		
		if df.shape[1] < 2:
			print(f"Warning: {layer_csv_file} does not have enough columns. Skipping this file.")
			continue
		
		#--------------
		# Extract the second value (K values)
		#--------------
		k_values = pd.to_numeric(df.iloc[:, 1], errors='coerce')       # Coerce invalid parsing to NaN

		# Filter out NaN values from k_values
		k_values = k_values.dropna()

		#print('k_values', k_values)
		#input('enter')

		if k_values.empty:
			print(f"Warning: No valid numeric data found in '{layer_csv_file}', skipping this file.")
			continue

		#--------------
		# Compute theta = 1/K
		#--------------
		theta = 1 / k_values
		
		#print('theta', theta)
		#input('enter')
		
		#theta = theta[ not pd.isnan(theta) and not pd.isinf(theta) ]              # Remove inf values if there exists
		
		theta = np.array(theta)
		
		theta = theta[ np.logical_and(np.logical_not(np.isnan(theta)), np.logical_not(np.isinf(theta))) ]
		
		#print('theta', theta)
		#input()
		#print('isnan', np.isnan(theta))
		#input()
		#print('not isnan', np.logical_and(np.logical_not(np.isnan(theta)), np.logical_not(np.isinf(theta)) )
		#input()
		#print('isinf', np.isinf(theta))
		#input()
		
		#theta = theta[not np.isnan(theta)]
		#theta = theta[not np.isinf(theta)]
		
		#--------------
		# Store theta values for plotting
		#--------------
		theta_values[layer_name] = theta
	else:
		print(f"Warning: {layer_csv_file} not found in directory.")

#----------------------------------------------------------
# Plot box plot for each layer
#----------------------------------------------------------
plt.figure(figsize=(7, 6))

#-------------------
# Draw horizontal lines
#-------------------
plt.axhline(y=0.5, color='g', linestyle='--', label='y = 0.5')
plt.axhline(y=1,   color='r', linestyle='--', label='y = 1')

#-------------------
# Add text labels at the y-axis
#-------------------
#plt.text(x= -0.83, y=0.5,  s='Gaussian',     fontsize=16,    verticalalignment='center',    color='g' )     #fontweight='bold'
#plt.text(x= -1.05,   y=1,  s='Exponential',  fontsize=16,    verticalalignment='center',    color='r' )     #fontweight='bold'

plt.text(x= -1.04, y=0.5,  s='Gaussian',       fontsize=16,    verticalalignment='center',    color='g' )   
plt.text(x= -1.32, y=1,    s='Exponential',    fontsize=16,    verticalalignment='center',    color='r' )

#-------------------
# Collect all theta values for box plot
#-------------------
boxplot_data = [theta_values[layer_name] for layer_name in layer_names if layer_name in theta_values]


#layer_labels = [f"Layer {i}" for i in range(len(boxplot_data))]
layer_labels = [f"{i}" for i in range(len(boxplot_data))]

medianprops=dict(color='black', linewidth=1.5)
plt.boxplot(boxplot_data, labels=layer_labels, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'), medianprops=medianprops)

#-------------------
#Axis labels and settings
#-------------------
ax = plt.gca()

#-------------------
# X axis
#-------------------

plt.xticks(fontsize=20)
#plt.xlabel('Layers', fontsize=18)
plt.xlabel('Layer', fontsize=21)      # this  is in the final version

#-------------------
# Y axis
#-------------------
ax.set_ylim([0, 3])
ax.set_yticks(np.arange(0, 3.5, 0.5))
plt.yticks(fontsize=20)
                            

plt.ylabel(r'Tail Parameter $\theta$', fontsize=21, labelpad=15)                    # labelpad=15     # fontsize=18       
ax.yaxis.set_label_coords(-0.12, 0.67)                                              # ax.yaxis.set_label_coords(-0.1, 0.6)

#-------------------
# Title
#-------------------
#uppercase_datasets = ['mnist']
#dataset_formatted = dataset.upper() if dataset.lower() in uppercase_datasets else dataset.capitalize()

if 'mnist' in dataset.lower():
	dataset_formatted = 'MNIST'

elif 'cifar100' in dataset.lower():
	dataset_formatted = 'CIFAR-100'

elif 'cifar10' in dataset.lower():
	dataset_formatted = 'CIFAR-10'
	
elif 'imagenette2' in dataset.lower():
	dataset_formatted = 'Imagenette2'
	

if 'resnet18' in model.lower():
	model_formatted = 'ResNet-18'
elif 'resnet50' in model.lower():
	model_formatted = 'ResNet-50'
elif 'vgg19' in model.lower():
	model_formatted = 'VGG-19'

plt.title(f'{dataset_formatted}  {model_formatted}', fontsize=24)
#plt.title('Histogram', fontsize=34, pad = 15)

#plt.title(f'{dataset} {model}', fontsize=15)
#plt.title(f'{dataset}_{model}_{seed}_{nquant}_{percentile}_{variant_flag}')

#-------------------
# Plot and Save
#-------------------
plot_file = os.path.join(box_plot_directory, f"{dataset}_{model}_{seed}_{nquant}_{percentile}_{variant_flag}_boxplot.png")
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

print(f"Plot saved at: {plot_file}")
#plt.show()

