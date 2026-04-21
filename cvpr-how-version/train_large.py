import os
import sys
import math
import random
import torch
import torchvision
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datasets
import models_large


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

def cross_entropy(Y,Yhat):
        N = Y.shape[0]
        return -torch.sum( Y * torch.log(Yhat + 0.0000001) ) / N


#----------------------------
# Check command arguments
#----------------------------
if (len(sys.argv)<4):
	print('Usage:')
	print('   train_large.py dataset model seed [nepoch batchsize learnrate pretrain allparam]')
	print('')
	print('  dataset:   imagenette2')
	print('  model:     resnet18 resnet50 vgg19')
	print('  seed:      seed for random number generator for reproducibility')
	print('  nepoch:    number of epochs  (default 20)')
	print('  batchsize: batch size        (default 128)')
	print('  learnrate: learning rate     (default 0.1)')
	print('  pretrain:  True use pretrained weights   False use random weights  (default True)')
	print('  allparam:  True finetune everything  False only last layer         (default False)')
	print('')
	sys.exit(1)
arg_dataset    =         sys.argv[1]
arg_model      =         sys.argv[2]
arg_seed       =     int(sys.argv[3])

arg_nepoch = 20
if (len(sys.argv)>4):
	arg_nepoch = int(sys.argv[4])

arg_batch_size = 128
if (len(sys.argv)>5):
	arg_batch_size = int(sys.argv[5])

arg_learn_rate = 0.01
if (len(sys.argv)>6):
	arg_learn_rate = float(sys.argv[6])

arg_pretrain = True
if (len(sys.argv)>7):
	arg_pretrain = BOOL(sys.argv[7])

arg_allparam = False
if (len(sys.argv)>8):
	arg_allparam = BOOL(sys.argv[8])

print('arg_dataset', arg_dataset)
print('arg_model', arg_model)
print('arg_seed', arg_seed)
print('arg_nepoch', arg_nepoch)
print('arg_batch_size', arg_batch_size)
print('arg_learn_rate', arg_learn_rate)
print('arg_pretrain', arg_pretrain)
print('arg_allparam', arg_allparam)


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
print(" Set seed")
print("-----------------------------------")
random.seed(arg_seed)         # Python seed
np.random.seed(arg_seed)      # Numpy seed
torch.manual_seed(arg_seed)   # Pytorch seed



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




print("-----------------------------------")
print(" Create model")
print("-----------------------------------")

if (arg_model == 'vgg19'):
	vgg19 = torchvision.models.vgg19(pretrained=True)
	vgg19 = vgg19.to(device)
	model = models_large.VGG19(vgg19, n_class)
	model = model.to(device)
if (arg_model == 'resnet18'):
	resnet18 = torchvision.models.resnet18(pretrained=True)
	resnet18 = resnet18.to(device)
	model = models_large.ResNet(resnet18, 'resnet18', n_class)
	model = model.to(device)
if (arg_model == 'resnet50'):
	resnet50 = torchvision.models.resnet50(pretrained=True)
	resnet50 = resnet50.to(device)
	model = models_large.ResNet(resnet50, 'resnet50', n_class)
	model = model.to(device)




print("-----------------------------------")
print(" Create output directory")
print("-----------------------------------")

# Create the output directory
mkdir('train')

outdir = 'train/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
mkdir(outdir)

# Save the command arguments
fargv = open(outdir+'/argv.txt','w')
fargv.write('arg_dataset %s\n' % arg_dataset)
fargv.write('arg_model %s\n' % arg_model)
fargv.write('arg_seed %d\n' % arg_seed)
fargv.write('arg_nepoch %d\n' % arg_nepoch)
fargv.write('arg_batch_size %d\n' % arg_batch_size)
fargv.write('arg_learn_rate %f\n' % arg_learn_rate)
fargv.write('arg_pretrain %s\n' % str(arg_pretrain))
fargv.write('arg_allparam %s\n' % str(arg_allparam))
fargv.close()

# Open the csv for accuracy/loss
fcsv = open(outdir+'/acc_loss.csv','w')
fcsv.write('epoch\tval_acc\tval_loss\ttrain_acc\ttrain_loss\t\n')
fcsv.flush()


print("-----------------------------------")
print(" Finetune the model")
print("-----------------------------------")

# batch sizes
batch_size = arg_batch_size
nBatch_train = int(math.ceil(n_train / batch_size))
nBatch_test  = int(math.ceil(n_test / batch_size))

# Define the loss function and the optimizer
#criterion = nn.CrossEntropyLoss()
if (arg_allparam):
	optimizer = optim.SGD(model.parameters(), lr=arg_learn_rate, momentum=0.9, weight_decay=5e-4)
else:
	optimizer = optim.SGD(model.fc.parameters(), lr=arg_learn_rate, momentum=0.9, weight_decay=5e-4)

# define a scheduler
# Set up one-cycle learning rate scheduler
#sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, arg_learn_rate, epochs=arg_nepoch, steps_per_epoch=nBatch_train)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# book-keeping for training/validation accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

#---------------
# For every epoch
#---------------
for epoch in range(arg_nepoch):

	print('epoch', epoch)

	#---------------
	# For every batch (training)
	#---------------
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0

	x_train,label_train = datasets.Shuffle(x_train,label_train)

	for batchno in range(nBatch_train):

		# Extract data and labels
		X,labels = datasets.Batch(x_train,label_train,batchno,batch_size,device)
		Y = F.one_hot(labels,n_class)

		#print('X', X)
		#input('enter')

		# Perform the step
		optimizer.zero_grad()
		Yhat = model(X)
		loss = cross_entropy(Y, Yhat)
		loss.backward()
		optimizer.step()
		sched.step()
		total_loss += loss.item()

		#print('train batchno', batchno, 'of', nBatch_train, 'loss', loss.item())

		#if (batchno%1000000==0):
		#	for j in range(Y.shape[0]):
		#		print('batchno', batchno, 'j', j)
		#		print('Y:', Y)
		#		print('Yhat:', Yhat)

		_, predicted = torch.max(Yhat.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	train_loss = total_loss / n_train
	train_accuracy = 100 * correct / total
	train_losses.append(train_loss)
	train_accuracies.append(train_accuracy)

	#---------------
	# For every batch (validation)
	#---------------
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for batchno in range(nBatch_test):
			#print('test batchno', batchno, 'of', nBatch_test)

			X,labels = datasets.Batch(x_test,label_test,batchno,batch_size,device)
			Y = F.one_hot(labels,n_class)
			Yhat = model(X)
			loss = cross_entropy(Y, Yhat)
			total_loss += loss.item()

			_, predicted = torch.max(Yhat.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	val_loss = total_loss / n_test
	val_accuracy = 100 * correct / total
	val_losses.append(val_loss)
	val_accuracies.append(val_accuracy)

	print(f'Epoch [{epoch+1}/{arg_nepoch}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
	sys.stdout.flush()

	#---------------
	# Save the accuracy / loss
	#---------------
	fcsv.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\t\n' % (epoch,val_accuracy,val_loss,train_accuracy,train_loss))
	fcsv.flush()

	#---------------
	# Save the model weights
	#---------------
	print('save %s/model.dict' % outdir)
	torch.save(model.state_dict(), '%s/model.dict' % outdir)
	print('saved')


fcsv.close()
print('Success!')
