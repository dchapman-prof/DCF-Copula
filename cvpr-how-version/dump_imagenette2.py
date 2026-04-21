import os
import sys
import cv2
import numpy as np
import torchvision

def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print('cannot make'+path)

def np_save(file,arr):
	print('np.save', file)
	np.save(file,arr)

#---------------------------------------
# The following code creates a train-loader
#  and a val loader
#---------------------------------------
def preprocess_imagenette2(img):

	# Get image dimensions
	height, width, _ = img.shape

	# Determine the scale factor to resize the image so that the shorter side is 256
	if width < height:
		new_width = 256
		new_height = 256 * height // width
	else:
		new_height = 256
		new_width = 256 * width // height

	# Resize the image using linear interpolation
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

	# Calculate the center point
	center_x = new_width // 2
	center_y = new_height // 2

	# Calculate the coordinates for the center crop
	left = (center_x - 112) // 2
	top = (center_y - 112) // 2
	right = left + 224
	bottom = top + 224

	# Perform the center crop
	img = img[top:bottom, left:right]

	# Normalize pixel values to range [0, 1]
	img = img.astype(np.float32) / 255.0

	return img


def normalize_imagenette2(rgb):

	# Normalize based on mean and stdev
	mean  = (0.485, 0.456, 0.406)
	stdev = (0.229, 0.224, 0.225)

	for c in range(3):
		rgb[c] = (rgb[c]-mean[c]) / stdev[c]
	return rgb




mkdir('datasets')


print("------------------------------------------------------")
print(" Load Imagenette2")
print("------------------------------------------------------")

print("-------------")
print(" Define the list of classes")
print("-------------")
nClass=10
class_name = ("tench",     "English springer", "cassette player", "chain saw", "church",    "French horn", "garbage truck", "gas pump",  "golf ball", "parachute")
class_id   = ("n01440764", "n02102040",        "n02979186",       "n03000684", "n03028079", "n03394916",   "n03417042",     "n03425413", "n03445777", "n03888257")

print("----------------------")
print(" Read the test directories")
print("----------------------")
test_class_paths      = [None]*nClass
test_class_img_names  = [None]*nClass
test_class_n          = [None]*nClass
test_class_labels     = [None]*nClass
for c in range(nClass):
	dir_path = 'imagenette2/val/'+str(class_id[c])
	img_names = os.listdir('imagenette2/val/'+str(class_id[c]))
	test_class_paths[c] = [dir_path+'/'+x for x in img_names]
	test_class_img_names[c]  = img_names
	#print(class_paths[c])
	test_class_n[c]      = len(test_class_paths[c])
	test_class_labels[c] = [c]*test_class_n[c]
	#input('enter')
#for i in range(len(class_nTest)):
#	class_nTest[i]=30
#input('enter')

print("-------------")
print(" How many test data ?")
print("-------------")
nTest = sum(test_class_n)
print('test_class_n', test_class_n)
print('nTest', nTest)

#input('enter')

print("-------------")
print(" Reformat paths into a 1D array")
print("-------------")
test_paths  = [None]*nTest
test_labels = np.zeros((nTest,), dtype=np.int32)
test_img_names = [None]*nTest
idx=0
for c in range(nClass):
	for i in range(test_class_n[c]):
		test_paths[idx]  = test_class_paths[c][i]
		test_labels[idx] = test_class_labels[c][i]
		test_img_names[idx] = test_class_img_names[c][i]
		idx+=1
	#print('test_paths', test_paths)
#input('enter')
#print('test_labels', test_labels)
#input('enter')
print("-------------")
print(" Read the testing images")
print("-------------")
test_imgs = [None]*nTest
test_imgs = np.zeros((nTest,3,224,224),dtype=np.float32)
for i in range(nTest):
	if (i%1000==0):
		print(i, 'of', nTest)
	path = test_paths[i]
	img = cv2.imread(path)
	img = preprocess_imagenette2(img)

	for ch in range(3):
		test_imgs[i,ch,:,:] = img[:,:,3-ch-1]
	test_imgs[i] = normalize_imagenette2(test_imgs[i])

#input('enter')


print("----------------------")
print(" Read the train directories")
print("----------------------")
train_class_paths      = [None]*nClass
train_class_img_names  = [None]*nClass
train_class_n          = [None]*nClass
train_class_labels     = [None]*nClass
for c in range(nClass):
	dir_path = 'imagenette2/train/'+str(class_id[c])
	img_names = os.listdir('imagenette2/train/'+str(class_id[c]))
	train_class_paths[c] = [dir_path+'/'+x for x in img_names]
	train_class_img_names[c]  = img_names
	#print(class_paths[c])
	train_class_n[c]      = len(train_class_paths[c])
	train_class_labels[c] = [c]*train_class_n[c]

#input('enter')
#for i in range(len(class_nTrain)):
#	class_nTrain[i]=30
#input('enter')

print("-------------")
print(" How many train data ?")
print("-------------")
nTrain = sum(train_class_n)
print('train_class_n', train_class_n)
print('nTrain', nTrain)

#input('enter')

print("-------------")
print(" Reformat paths into a 1D array")
print("-------------")
train_paths  = [None]*nTrain
train_labels = np.zeros((nTrain,), dtype=np.int32)
train_img_names = [None]*nTrain
idx=0
for c in range(nClass):
	for i in range(train_class_n[c]):
		train_paths[idx]  = train_class_paths[c][i]
		train_labels[idx] = train_class_labels[c][i]
		train_img_names[idx] = train_class_img_names[c][i]
		idx+=1

#print('train_paths', train_paths)
#input('enter')
#print('train_labels', train_labels)
#input('enter')

print("-------------")
print(" Read the training images")
print("-------------")
train_imgs = np.zeros((nTrain,3,224,224),dtype=np.float32)
for i in range(nTrain):
	if (i%1000==0):
		print(i, 'of', nTrain)
	path = train_paths[i]
	img = cv2.imread(path)
	img = preprocess_imagenette2(img)
	for ch in range(3):
		train_imgs[i,ch,:,:] = img[:,:,3-ch-1]
	train_imgs[i] = normalize_imagenette2(train_imgs[i])

#input('enter')


print("------------------------------------------------------")
print(" Save Imagenette2")
print("------------------------------------------------------")

mkdir('datasets/imagenette2')

np_save('datasets/imagenette2/x_train.npy', train_imgs)
np_save('datasets/imagenette2/label_train.npy', train_labels)
np_save('datasets/imagenette2/x_test.npy', test_imgs)
np_save('datasets/imagenette2/label_test.npy', test_labels)

print('save datasets/imagenette2/class_name.txt')
f=open('datasets/imagenette2/class_name.txt','w')
for name in class_name:
	f.write(name + '\n')
f.close()


print('Done!')

