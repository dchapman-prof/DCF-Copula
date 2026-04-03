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
csv_directory = os.path.join(base_directory, f"{dataset}_{model}_{seed}_{nquant}_{percentile}_{variant_flag}/param/")


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

#-----------------------------
# Plot box plot for each layer
#-----------------------------
plt.figure(figsize=(10, 6))

# Draw horizontal lines
plt.axhline(y=0.5, color='r', linestyle='--', label='y = 0.5')
plt.axhline(y=1, color='b', linestyle='--', label='y = 1')

# Add text labels at the y-axis
plt.text(x= -0.1, y=0.5, s='Gaussian',    fontsize=12,    verticalalignment='center', color='r')
plt.text(x= -0.3, y=1,   s='Exponential', fontsize=12, verticalalignment='center',    color='b')

# Collect all theta values for box plot
boxplot_data = [theta_values[layer_name] for layer_name in layer_names if layer_name in theta_values]


layer_labels = [f"Layer {i}" for i in range(len(boxplot_data))]

plt.boxplot(boxplot_data, labels=layer_labels, patch_artist=True, boxprops=dict(facecolor='yellow', color='black'))
ax = plt.gca()
ax.set_ylim([0, 5])

#plt.xlabel('Layers')
plt.ylabel(r'$\theta = 1/K$')
plt.title(f'Box Plot of Theta Values for {dataset}_{model}_{seed}_{nquant}_{percentile}_{variant_flag}')
plt.show()

