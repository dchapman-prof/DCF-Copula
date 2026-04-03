import os
import sys
import math
import torch
import torchvision
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datasets
import model_loader


def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print('cannot make'+path)

def np_save(file,arr):
	print('np.save', file)
	np.save(file,arr)

def BOOL(str):
	if (str=='False'):
		return False
	elif (str=='True'):
		return True
	else:
		print('ERROR: expected True or False')
		sys.exit(1)

#----------------------------
# Check command arguments
#----------------------------
if (len(sys.argv)<5):
	print('Usage')
	print('')
	print('   python3 extract_feat.py dataset.py model.py seed batchsize')
	print('')
	print('  dataset:   cifar10 cifar100 imagenette2 mnist')
	print('  model:     resnet18 resnet50 vgg19')
	print('  seed:      integer seed for which model to load')
	print('  batchsize: how many images to process at once')
	sys.exit(1)
arg_dataset    =         sys.argv[1]
arg_model      =         sys.argv[2]
arg_seed       =     int(sys.argv[3])
arg_batch_size =     int(sys.argv[4])

print('arg_dataset', arg_dataset)
print('arg_model', arg_model)
print('arg_seed', arg_seed)
print('arg_batch_size', arg_batch_size)

print("-----------------------------------")
print(" Obtain accelerator")
print("-----------------------------------")
sys.stdout.flush()
if torch.backends.mps.is_available():
    print('use mps')
    device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    device = torch.device("cuda")
else:
    print('use cpu')
    device = torch.device("cpu")



print("-----------------------------------")
print(" Load dataset")
print("-----------------------------------")

x_train,label_train,x_test,label_test,class_name = datasets.Load(arg_dataset)

n_train = x_train.shape[0]
n_test  = x_test.shape[0]
n_class = len(class_name)
print('n_train', n_train)
print('n_test', n_test)
print('n_class', n_class)

batch_size = arg_batch_size
nBatch_train = int(math.ceil(n_train / batch_size))
nBatch_test  = int(math.ceil(n_test / batch_size))
print('batch_size', batch_size)
print('nBatch_train', nBatch_train)
print('nBatch_test', nBatch_test)


print("-----------------------------------")
print(" Load model")
print("-----------------------------------")

model = model_loader.load_trained_model(arg_dataset, arg_model, arg_seed, device)


print("-----------------------------------")
print(" Make the output directories")
print("-----------------------------------")

mkdir('features')
featdir = 'features/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
mkdir(featdir)


print("-----------------------------------")
print(" Extract the features")
print("-----------------------------------")

def SafeGet(obj,attr):
	if (hasattr(obj,attr)):
		return getattr(obj,attr)
	else:
		return None

def SafeShape(shape,path):
	if (shape is None):
		return
	print('Write', path)
	f = open(path,'w')
	for i in range(len(shape)):
		f.write('%d ' % shape[i])
	f.write('float32\n')
	f.close()

def SafeOpen(f,path,flag):
	if (f is None):
		print('Open', path)
		f = open(path,flag)
	return f

def SafeClose(f):
	if (f is not None):
		f.close()

def SafeAppend(X,shape,f,path):
	if X is None:
		return (shape,f)
	f = SafeOpen(f,path,'wb')
	X_np = X.detach().cpu().numpy()
	if shape is None:
		shape = list(X_np.shape)
	else:
		shape[0] += X_np.shape[0]
	bytes = X_np.astype(np.float32).tobytes()
	f.write(bytes)
	return (shape,f)


#---------------
# For every batch (training)
#---------------
fX_train = None
fA_train = None
fB_train = None
fC_train = None
fD_train = None
fE_train = None
fX_test = None
fA_test = None
fB_test = None
fC_test = None
fD_test = None
fE_test = None
fX_train_shape = None
fA_train_shape = None
fB_train_shape = None
fC_train_shape = None
fD_train_shape = None
fE_train_shape = None
fX_test_shape = None
fA_test_shape = None
fB_test_shape = None
fC_test_shape = None
fD_test_shape = None
fE_test_shape = None


model.eval()
with torch.no_grad():
	for batchno in range(nBatch_train):
	#for batchno in range(2):

		print('train batch', batchno, 'of', nBatch_train)

		# Extract data and labels
		X,labels = datasets.Batch(x_train,label_train,batchno,batch_size,device)
		Y = F.one_hot(labels,n_class)
		Yhat = model(X)

		# Extract the data
		resX = SafeGet(model,'resX')
		resA = SafeGet(model,'resA')
		resB = SafeGet(model,'resB')
		resC = SafeGet(model,'resC')
		resD = SafeGet(model,'resD')
		resE = SafeGet(model,'resE')

		# Append the data
		fX_train_shape,fX_train = SafeAppend(resX,fX_train_shape,fX_train,featdir+'/X_train.bin')
		fA_train_shape,fA_train = SafeAppend(resA,fA_train_shape,fA_train,featdir+'/A_train.bin')
		fB_train_shape,fB_train = SafeAppend(resB,fB_train_shape,fB_train,featdir+'/B_train.bin')
		fC_train_shape,fC_train = SafeAppend(resC,fC_train_shape,fC_train,featdir+'/C_train.bin')
		fD_train_shape,fD_train = SafeAppend(resD,fD_train_shape,fD_train,featdir+'/D_train.bin')
		fE_train_shape,fE_train = SafeAppend(resE,fE_train_shape,fE_train,featdir+'/E_train.bin')

# Close the files
SafeClose(fX_train)
SafeClose(fA_train)
SafeClose(fB_train)
SafeClose(fC_train)
SafeClose(fD_train)
SafeClose(fE_train)

# Write the shape
SafeShape(fX_train_shape, featdir+'/X_train.bin.txt')
SafeShape(fA_train_shape, featdir+'/A_train.bin.txt')
SafeShape(fB_train_shape, featdir+'/B_train.bin.txt')
SafeShape(fC_train_shape, featdir+'/C_train.bin.txt')
SafeShape(fD_train_shape, featdir+'/D_train.bin.txt')
SafeShape(fE_train_shape, featdir+'/E_train.bin.txt')


#---------------
# For every batch (validation)
#---------------
model.eval()
with torch.no_grad():
	for batchno in range(nBatch_test):
	#for batchno in range(2):

		print('test batch', batchno, 'of', nBatch_test)

		# Extract data and labels
		X,labels = datasets.Batch(x_test,label_test,batchno,batch_size,device)
		Y = F.one_hot(labels,n_class)
		Yhat = model(X)

		# Extract the data
		resX = SafeGet(model,'resX')
		resA = SafeGet(model,'resA')
		resB = SafeGet(model,'resB')
		resC = SafeGet(model,'resC')
		resD = SafeGet(model,'resD')
		resE = SafeGet(model,'resE')


		# Append the data
		fX_test_shape,fX_test = SafeAppend(resX,fX_test_shape,fX_test,featdir+'/X_test.bin')
		fA_test_shape,fA_test = SafeAppend(resA,fA_test_shape,fA_test,featdir+'/A_test.bin')
		fB_test_shape,fB_test = SafeAppend(resB,fB_test_shape,fB_test,featdir+'/B_test.bin')
		fC_test_shape,fC_test = SafeAppend(resC,fC_test_shape,fC_test,featdir+'/C_test.bin')
		fD_test_shape,fD_test = SafeAppend(resD,fD_test_shape,fD_test,featdir+'/D_test.bin')
		fE_test_shape,fE_test = SafeAppend(resE,fE_test_shape,fE_test,featdir+'/E_test.bin')

# Close the files
SafeClose(fX_test)
SafeClose(fA_test)
SafeClose(fB_test)
SafeClose(fC_test)
SafeClose(fD_test)
SafeClose(fE_test)

# Write the shape
SafeShape(fX_test_shape, featdir+'/X_test.bin.txt')
SafeShape(fA_test_shape, featdir+'/A_test.bin.txt')
SafeShape(fB_test_shape, featdir+'/B_test.bin.txt')
SafeShape(fC_test_shape, featdir+'/C_test.bin.txt')
SafeShape(fD_test_shape, featdir+'/D_test.bin.txt')
SafeShape(fE_test_shape, featdir+'/E_test.bin.txt')

print('Done\n')

