import cv2
import numpy as np
import os

layers = ['X', 'A', 'B', 'C', 'D']
FA = [15, 14, 15, 72, 206]
FB = [19, 40, 85, 245, 484]
series = ['leg', 'fou', 'hist']

# Initialize the big image container
bigimg = None

for j in range(3):
	ser = series[j]
	for i in range(5):
		lay = layers[i]
		a   = FA[i]
		b   = FB[i]
        
        # Construct the file path
	filename = f"{ser}_{a}_{b}.png"
	folder = f"imagenette2_resnet18_0_{lay}_f_4_m_11_b_10"
	path = os.path.join("pit_test", folder, filename)

	print(f"Reading: {path}")
	img = cv2.imread(path)

	# Check if the image is successfully loaded
	if img is None:
	    print(f"Error: Could not read image at {path}")
	    continue

	# Initialize the big image container if not already done
	if bigimg is None:
	    sY, sX, ch = img.shape
	    bigimg = np.zeros((5 * sY, 3 * sX, ch), dtype=img.dtype)

	# Place the image in the correct position
	bigimg[i * sY:(i + 1) * sY, j * sX:(j + 1) * sX] = img

# Save the final combined image
if bigimg is not None:
	cv2.imwrite('make_fig.png', bigimg)
	print("Figure saved as 'make_fig.png'")
else:
	print("No images were loaded; 'make_fig.png' was not created.")

