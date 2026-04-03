import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
#import torchvision.models as models
import models_small
import models_large

def load_trained_model(my_dataset, my_model, my_seed, my_device, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):

	n_chan = 3
	if (my_dataset=='mnist'):
		n_chan = 1

	n_classes = 10
	if (my_dataset=='cifar100'):
		n_classes=100

	if (my_dataset=='imagenette2'):
		if (my_model == 'vgg19'):
			vgg19 = torchvision.models.vgg19(pretrained=False)
			vgg19 = vgg19.to(my_device)
			model = models_large.VGG19(vgg19, n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		elif (my_model == 'resnet18'):
			resnet18 = torchvision.models.resnet18(pretrained=False)
			resnet18 = resnet18.to(my_device)
			model = models_large.ResNet(resnet18, 'resnet18', n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		elif (my_model == 'resnet50'):
			resnet50 = torchvision.models.resnet50(pretrained=False)
			resnet50 = resnet50.to(my_device)
			model = models_large.ResNet(resnet50, 'resnet50', n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		else:
			print('unexpected model', my_model)
			sys.exit(1)

	elif (my_dataset in ('mnist', 'cifar10', 'cifar100')):
		if (my_model == 'resnet18'):
			model = models_small.ResNet18(n_chan, n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		elif (my_model == 'resnet50'):
			model = models_small.ResNet50(n_chan, n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		elif (my_model == 'vgg19'):
			model = models_small.VGG('VGG19', n_chan, n_classes, thresh_arrs, zero_chance, thresh_mode)
			model = model.to(my_device)
		else:
			print('unexpected model', my_model)
			sys.exit(1)


	else:
		print('ERROR unknown dataset', my_dataset)
		sys.exit(1)

	dict_path = 'train/%s_%s_%d/model.dict' % (my_dataset, my_model, my_seed)
	print('load', dict_path)
	model.load_state_dict(torch.load(dict_path))

	return model
