import os
import sys
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#   Models for full size images (ImageNet)
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#   ResNet (Full Size Images)
#---------------------------------------------------------------------------

def veclen(X):
	return torch.sqrt( torch.sum(X*X) )

#def applyThresh(z,thresh):
#	if thresh is not None:
#		if not isinstance(thresh,float):
#			thresh = torch.reshape(thresh, (1,z.shape[1],1,1)).repeat((z.shape[0],1,z.shape[2],z.shape[3]))
#		z_filter = z * (z>thresh).float()
#		z_scale = veclen(z) / (veclen(z_filter) + 0.00001)
#		z = z_filter * z_scale
#	return z
def applyThresh(z,thresh,thresh_mode):
	if thresh is not None:
		if not isinstance(thresh,float):
			thresh = torch.reshape(thresh, (1,z.shape[1],1,1)).repeat((z.shape[0],1,z.shape[2],z.shape[3]))
		if thresh_mode=='low':
			z_filter = z * (z>thresh).float()
		if thresh_mode=='high':
			z_filter = z * (z<=thresh).float()			
		if thresh_mode=='clip':
			z_clip   = thresh * (z>thresh).float()
			z_filter = z * (z<=thresh).float()
			z_filter = z_filter + z_clip
		z_scale = veclen(z) / (veclen(z_filter) + 0.00001)
		z = z_filter * z_scale
	return z

def applyRandomZeros(z,chance):
	if chance > 0.0:
		random = torch.rand(z.shape, device=z.device)
		#print('random')
		#print(random)
		#input()

		mask = (random>chance).float()
		#print('mask')
		#print(mask)
		#input()
		
		z_filter = z * mask
		z_scale = veclen(z) / (veclen(z_filter) + 0.00001)
		z = z_filter * z_scale
	return z

class ResNet(nn.Module):
	def __init__(self, base_model, arg_model, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
		super(ResNet, self).__init__()

		# Establish the base_model as renset18
		self.base_model = base_model
		self.base_model.fc = nn.Identity()
		self.softmax = nn.Softmax(dim=1)
		self.zero_chance = zero_chance

		if (arg_model=='resnet18'):
			emb_dim = 512
		elif (arg_model=='resnet50'):
			emb_dim = 2048

		# Fully connected BEFORE average pooling
		self.fc = nn.Conv2d(emb_dim, num_classes, kernel_size=(1,1), stride=(1,1), padding=(0,0))

		# Keep track of the threshold tensors
		self.thresh_mode = thresh_mode
		self.threshA = None
		self.threshB = None
		self.threshC = None
		self.threshD = None
		self.threshE = None
		if thresh_arrs is not None:
			self.threshX = thresh_arrs[0]
			self.threshA = thresh_arrs[1]
			self.threshB = thresh_arrs[2]
			self.threshC = thresh_arrs[3]
			self.threshD = thresh_arrs[4]

	def forward(self, x):

		# Apply the basic model (deprecated)
		#out = self.base_model(x)

		# Apply the basic model manually
		x = self.base_model.conv1(x)
		self.resXconv1 = x     # store the result

		x = self.base_model.bn1(x)
		self.resXbn1 = x       # store the result

		x = self.base_model.relu(x)
		self.resXrelu1 = x      # store the result

		x = self.base_model.maxpool(x)

		# Apply layer 1
		a = self.base_model.layer1(x)

		# Threshold layer A
		a = applyThresh(a, self.threshA, self.thresh_mode)
		a = applyRandomZeros(a,self.zero_chance)

		# Apply layer 2
		b = self.base_model.layer2(a)

		# Threshold layer B
		b = applyThresh(b, self.threshB, self.thresh_mode)
		b = applyRandomZeros(b,self.zero_chance)

		# Apply layer 3
		c = self.base_model.layer3(b)


		# Threshold layer C
		c = applyThresh(c, self.threshC, self.thresh_mode)
		c = applyRandomZeros(c,self.zero_chance)

		# Apply layer number 4
		d = self.base_model.layer4(c)

		# Threshold layer D
		d = applyThresh(d, self.threshD, self.thresh_mode)
		d = applyRandomZeros(d,self.zero_chance)

		# Downsample the final classifier
		cam = self.fc(d)

		# Pool + Softmax classifier
		out = self.base_model.avgpool(cam)
		out = torch.flatten(out, 1)
		out = self.softmax(out)

		# Save the other outputs as state
		self.resX = x
		self.resA = a
		self.resB = b
		self.resC = c
		self.resD = d
		self.resCam = cam

		# return the output
		return out

#---------------------------------------------------------------------------
#   VGG19 (Full Size Images)
#---------------------------------------------------------------------------

class VGG19(nn.Module):
	def __init__(self, base_model, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
		super(VGG19, self).__init__()

		self.base_model = base_model
		self.num_classes = num_classes
		self.softmax = nn.Softmax(dim=1)

		# Keep track of the threshold tensors
		self.thresh_mode = thresh_mode
		self.threshA = None
		self.threshB = None
		self.threshC = None
		self.threshD = None
		self.threshE = None
		if thresh_arrs is not None:
			self.threshA = thresh_arrs[0]
			self.threshB = thresh_arrs[1]
			self.threshC = thresh_arrs[2]
			self.threshD = thresh_arrs[3]
			self.threshE = thresh_arrs[4]

		# Keep track of the probability of random zeros
		self.zero_chance = zero_chance

		#
		# VGG19 layer 1   64 filters
		#
		self.layer1_conv1 = self.base_model.features[0]
		self.layer1_relu1 = self.base_model.features[1]
		self.layer1_conv2 = self.base_model.features[2]
		self.layer1_relu2 = self.base_model.features[3]
		self.layer1_pool  = self.base_model.features[4]

		#
		# VGG19 layer 2   128 filters
		#
		self.layer2_conv1 = self.base_model.features[5]
		self.layer2_relu1 = self.base_model.features[6]
		self.layer2_conv2 = self.base_model.features[7]
		self.layer2_relu2 = self.base_model.features[8]
		self.layer2_pool  = self.base_model.features[9]

		#
		# VGG19 layer 3   256 filters
		#
		self.layer3_conv1 = self.base_model.features[10]
		self.layer3_relu1 = self.base_model.features[11]
		self.layer3_conv2 = self.base_model.features[12]
		self.layer3_relu2 = self.base_model.features[13]
		self.layer3_conv3 = self.base_model.features[14]
		self.layer3_relu3 = self.base_model.features[15]
		self.layer3_conv4 = self.base_model.features[16]
		self.layer3_relu4 = self.base_model.features[17]
		self.layer3_pool  = self.base_model.features[18]

		#
		# VGG19 layer 4   512 filters
		#
		self.layer4_conv1 = self.base_model.features[19]
		self.layer4_relu1 = self.base_model.features[20]
		self.layer4_conv2 = self.base_model.features[21]
		self.layer4_relu2 = self.base_model.features[22]
		self.layer4_conv3 = self.base_model.features[23]
		self.layer4_relu3 = self.base_model.features[24]
		self.layer4_conv4 = self.base_model.features[25]
		self.layer4_relu4 = self.base_model.features[26]
		self.layer4_pool  = self.base_model.features[27]

		#
		# VGG19 layer 5   512 filters
		#
		self.layer5_conv1 = self.base_model.features[28]
		self.layer5_relu1 = self.base_model.features[29]
		self.layer5_conv2 = self.base_model.features[30]
		self.layer5_relu2 = self.base_model.features[31]
		self.layer5_conv3 = self.base_model.features[32]
		self.layer5_relu3 = self.base_model.features[33]
		self.layer5_conv4 = self.base_model.features[34]
		self.layer5_relu4 = self.base_model.features[35]
		self.layer5_pool  = self.base_model.features[36]

		#
		# classification layer
		#
		self.class_fc1   = self.base_model.classifier[0]
		self.class_relu1 = self.base_model.classifier[1]
		self.class_drop1 = self.base_model.classifier[2]
		self.class_fc2   = self.base_model.classifier[3]
		self.class_relu2 = self.base_model.classifier[4]
		self.class_drop2 = self.base_model.classifier[5]

		#
		# Final fully connected layer
		#
		self.fc = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

	def forward(self,x):

		#
		# VGG19 layer 1   64 filters
		#
		a = self.layer1_conv1(x)
		a = self.layer1_relu1(a)
		a = self.layer1_conv2(a)
		a = self.layer1_relu2(a)

		# Threshold layer A
		a = applyThresh(a, self.threshA, self.thresh_mode)
		a = applyRandomZeros(a,self.zero_chance)
		self.resA = a

		a = self.layer1_pool(a)

		#
		# VGG19 layer 2   128 filters
		#
		b = self.layer2_conv1(a)
		b = self.layer2_relu1(b)
		b = self.layer2_conv2(b)
		b = self.layer2_relu2(b)

		# Threshold layer B
		b = applyThresh(b, self.threshB, self.thresh_mode)
		b = applyRandomZeros(b,self.zero_chance)
		self.resB = b

		b = self.layer2_pool(b)

		#
		# VGG19 layer 3   256 filters
		#
		c = self.layer3_conv1(b)
		c = self.layer3_relu1(c)
		c = self.layer3_conv2(c)
		c = self.layer3_relu2(c)
		c = self.layer3_conv3(c)
		c = self.layer3_relu3(c)
		c = self.layer3_conv4(c)
		c = self.layer3_relu4(c)


		# Threshold layer C
		c = applyThresh(c, self.threshC, self.thresh_mode)
		c = applyRandomZeros(c,self.zero_chance)
		self.resC = c

		c = self.layer3_pool(c)

		#
		# VGG19 layer 4   512 filters
		#
		d = self.layer4_conv1(c)
		d = self.layer4_relu1(d)
		d = self.layer4_conv2(d)
		d = self.layer4_relu2(d)
		d = self.layer4_conv3(d)
		d = self.layer4_relu3(d)
		d = self.layer4_conv4(d)
		d = self.layer4_relu4(d)

		# Threshold layer D
		d = applyThresh(d, self.threshD, self.thresh_mode)
		d = applyRandomZeros(d,self.zero_chance)
		self.resD = d

		d = self.layer4_pool(d)

		#
		# VGG19 layer 5   512 filters
		#
		e = self.layer5_conv1(d)
		e = self.layer5_relu1(e)
		e = self.layer5_conv2(e)
		e = self.layer5_relu2(e)
		e = self.layer5_conv3(e)
		e = self.layer5_relu3(e)
		e = self.layer5_conv4(e)
		e = self.layer5_relu4(e)

		# Threshold layer E
		e = applyThresh(e, self.threshE, self.thresh_mode)
		e = applyRandomZeros(e,self.zero_chance)
		self.resE = e

		e = self.layer5_pool(e)

		#
		# classification layer
		#
		f = self.base_model.avgpool(e)
		f = torch.flatten(f, 1)
		f = self.class_fc1(f)
		f = self.class_relu1(f)
		f = self.class_drop1(f)
		f = self.class_fc2(f)
		f = self.class_relu2(f)
		f = self.class_drop2(f)
		self.resF = f

		#
		# Final fully connected layer (custom)
		#
		out = self.fc(f)
		out = self.softmax(out)

		return out

def cross_entropy(Y,Yhat):
	#print('Y.shape', Y.shape)
	#print('Yhat.shape', Yhat.shape)
	#input('enter')
	N = Y.shape[0]
	#C = Y.shape[1]
	#for n in range(N):
	#	for c in range(C):
	#		print('img', n, 'Y', Y[n,c].item(), 'Yhat', Yhat[n,c].item())
	#	input('enter')
	return -torch.sum( Y * torch.log(Yhat + 0.0000001) ) / N




