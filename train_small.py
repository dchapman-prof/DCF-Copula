#-----------------------------------
# train_small.py modified from
#
#  https://github.com/kuangliu/pytorch-cifar/tree/master
#
# The following license pertains specifically to this file
#
# MIT License
#
# Copyright (c) 2017 liukuang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#-----------------------------------

import os
import sys
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import models_small

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print('cannot make'+path)

def np_save(file,arr):
	print('np.save', file)
	np.save(file,arr)


#-----------------------------
# Read Command Arguments
#-----------------------------
if (len(sys.argv)<3):
	print('Usage:')
	print('')
	print('   python3 train_small.py dataset model seed [nepoch] [lr]')
	print('')
	print(' dataset:   cifar10 cifar100 mnist')
	print(' model:     resnet18 resnet50 vgg19')
	print(' seed:      integer seed for random generator')
	print(' nepoch:    number of epochs (default 200)')
	print(' lr:        learning rate (default 0.1)')
	print('')
	sys.exit(1)

arg_dataset =     sys.argv[1]
if (arg_dataset not in ('cifar10', 'cifar100', 'mnist')):
	print('Unexpected dataset', arg_dataset, 'expected cifar10 cifar100 or mnist')
	sys.exit(1)

arg_model   =     sys.argv[2]
if (arg_model not in ('resnet18', 'resnet50', 'vgg19')):
	print('Unexpected model', arg_model, 'expected resnet18 resnet50 or vgg19')
	sys.exit(1)

arg_seed    = int(sys.argv[3])

arg_nepoch = 200
if (len(sys.argv)>4):
	arg_nepoch = int(sys.argv[4])

arg_lr = 0.1
if (len(sys.argv)>5):
	arg_lr = float(sys.argv[5])

print('arg_dataset', arg_dataset)
print('arg_model  ', arg_model)
print('arg_seed   ', arg_seed)
print('arg_nepoch ', arg_nepoch)
print('arg_lr     ', arg_lr)


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
print(" Prepare dataset")
print("-----------------------------------")

if (arg_dataset in ('cifar10', 'cifar100')):
	transform_augment = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_simple = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	n_chan = 3

if (arg_dataset == 'cifar10'):
	trainset_augment = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augment)
	trainloader_augment  = torch.utils.data.DataLoader(trainset_augment, batch_size=128, shuffle=True, num_workers=2)

	trainset     = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_simple)
	trainloader   = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_simple)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if (arg_dataset == 'cifar100'):
	trainset_augment = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_augment)
	trainloader_augment  = torch.utils.data.DataLoader(trainset_augment, batch_size=128, shuffle=True, num_workers=2)

	trainset     = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_simple)
	trainloader   = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_simple)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ( \
		"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
		"bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
		"chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
		"dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
		"lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
		"mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
		"plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
		"rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
		"squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
		"train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm")

if (arg_dataset == 'mnist'):

	transform_augment = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
	])

	transform_simple = transforms.Compose([
		transforms.CenterCrop(32),
		transforms.ToTensor(),
	])
	n_chan = 1

	trainset_augment = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augment)
	trainloader_augment  = torch.utils.data.DataLoader(trainset_augment, batch_size=128, shuffle=True, num_workers=2)

	trainset     = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_simple)
	trainloader   = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

	testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_simple)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
n_classes = len(classes)



print("-----------------------------------")
print(" Save dataset")
print("-----------------------------------")

print('extract training batches to CPU')
n_train=0
train_data_batch   = []
train_label_batch = []
for batch_idx, (inputs, targets) in enumerate(trainloader):
	#print('train batch %d' % batch_idx)
	train_data_batch.append( inputs.detach().cpu().numpy() )
	train_label_batch.append( targets.detach().cpu().numpy() )
	n_train += inputs.shape[0]
n_batch_train = batch_idx

print('extract testing batches to CPU')
n_test=0
test_data_batch = []
test_label_batch = []
for batch_idx, (inputs, targets) in enumerate(testloader):
	#print('test batch %d' % batch_idx)
	test_data_batch.append( inputs.detach().cpu().numpy() )
	test_label_batch.append( targets.detach().cpu().numpy() )
	n_test += inputs.shape[0]
n_batch_test = batch_idx

print('compile training batches into a single numpy array')
chan = train_data_batch[0].shape[1]
sY   = train_data_batch[0].shape[2]
sX   = train_data_batch[0].shape[2]
train_data = np.zeros( (n_train,chan,sY,sX), dtype=np.float32)
train_labels = np.zeros( (n_train,), dtype=np.int64)
sidx = 0
eidx = 0
for batch_idx in range(n_batch_train):
	inputs = train_data_batch[batch_idx]
	labels = train_label_batch[batch_idx]
	sidx = eidx
	eidx = sidx + inputs.shape[0]
	train_data[sidx:eidx] = inputs
	train_labels[sidx:eidx] = labels

print('compile testing batches into a single numpy array')
test_data = np.zeros( (n_test,chan,sY,sX), dtype=np.float32)
test_labels = np.zeros( (n_test,), dtype=np.int64)
sidx = 0
eidx = 0
for batch_idx in range(n_batch_test):
	inputs = test_data_batch[batch_idx]
	labels = test_label_batch[batch_idx]
	sidx = eidx
	eidx = sidx + inputs.shape[0]
	test_data[sidx:eidx] = inputs
	test_labels[sidx:eidx] = labels

print('save data to numpy arrays')

mkdir('datasets')

datadir = 'datasets/'+arg_dataset
mkdir(datadir)

np_save(datadir+'/x_train.npy', train_data)
np_save(datadir+'/x_test.npy',  test_data)
np_save(datadir+'/label_train.npy', train_labels)
np_save(datadir+'/label_test.npy', test_labels)
fout = open(datadir+'/class_name.txt', 'w')
for name in classes:
	fout.write(name+'\n')
fout.close()



print("-----------------------------------")
print(" Construct Model")
print("-----------------------------------")

if (arg_model == 'resnet18'):
	net = models_small.ResNet18(n_chan, n_classes)
elif (arg_model == 'resnet50'):
	net = models_small.ResNet50(n_chan, n_classes)
elif (arg_model == 'vgg19'):
	net = models_small.VGG('VGG19', n_chan, n_classes)
else:
	print('unexpected model', arg_model)
	sys.exit(1)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



print("-----------------------------------")
print(" Create training directory and book-keeping files")
print("-----------------------------------")

mkdir('train')

traindir = 'train/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
mkdir(traindir)

facc = open(traindir + '/acc_loss.csv', 'w')
facc.write('epoch\tval_acc\tval_loss\ttrain_acc\ttrain_loss\t\n')

fargv = open(traindir + '/argv.txt', 'w')
fargv.write('arg_dataset %s\n' % arg_dataset)
fargv.write('arg_model %s\n' % arg_model)
fargv.write('arg_seed %d\n' % arg_seed)
fargv.write('arg_nepoch %d\n' % arg_nepoch)
fargv.write('arg_lr %f\n' % arg_lr)
fargv.close()



print("-----------------------------------")
print(" Train the Model")
print("-----------------------------------")

optimizer = optim.SGD(net.parameters(), lr=arg_lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def cross_entropy_loss(Yhat,targets):
    N = Yhat.shape[0]
    C = Yhat.shape[1]
    Y = F.one_hot(targets, C)
    return -torch.sum( Y * torch.log(Yhat + 0.0000001) ) / N


best_acc = 0  # best test accuracy


#--------------------------------
# For every epoch
#--------------------------------
for epoch in range(arg_nepoch):


	#--------------------
	# Training step
	#--------------------
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader_augment):
		#print('Epoch %d batch %d of %d' % (epoch, batch_idx, len(trainloader_augment)) )
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		#loss = criterion(outputs, targets)
		loss = cross_entropy_loss(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
	scheduler.step()

	train_loss_ = train_loss/(batch_idx)
	train_acc_  = 100.*correct/total
	print('Train Epoch %d Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss_, train_acc_, correct, total))


	#--------------------
	# Testing step
	#--------------------
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			#print('Epoch %d batch %d of %d' % (epoch, batch_idx, len(testloader)) )
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			#loss = criterion(outputs, targets)
			loss = cross_entropy_loss(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	test_loss_ = test_loss/(batch_idx)
	test_acc_  = 100.*correct/total
	print('Test  Epoch %d Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss_, test_acc_, correct, total))

	facc.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\t\n' % (epoch,test_acc_,test_loss_,train_acc_,train_loss_))
	facc.flush()

facc.close()


print("-----------------------------------")
print(" Save the weights")
print("-----------------------------------")
torch.save(net.state_dict(), '%s/model.dict' % traindir)

print('Success!')




#---------------------
# Deprecated
#---------------------
'''
	#--------------------
	# Save checkpoint.
	#--------------------
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
		}
		mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt.pth')
		best_acc = acc
'''
