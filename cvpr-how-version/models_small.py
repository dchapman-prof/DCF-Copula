#-----------------------------------
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
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#   Models for small size images (CIFAR10, CIFAR100, MNIST)
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


def veclen(X):
	return torch.sqrt( torch.sum(X*X) )

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

#---------------------------------------------------------------------------
#   ResNet (Small Size Images)
#---------------------------------------------------------------------------

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512*block.expansion, num_classes)
		self.softmax = nn.Softmax()

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

		# Keep track of the random zeros
		self.zero_chance = zero_chance


	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		self.resX = out
		out = self.layer1(out)
		out = applyThresh(out, self.threshA, self.thresh_mode)
		out = applyRandomZeros(out, self.zero_chance)
		self.resA = out
		out = self.layer2(out)
		out = applyThresh(out, self.threshB, self.thresh_mode)
		out = applyRandomZeros(out, self.zero_chance)
		self.resB = out
		out = self.layer3(out)
		out = applyThresh(out, self.threshC, self.thresh_mode)
		out = applyRandomZeros(out, self.zero_chance)
		self.resC = out
		out = self.layer4(out)
		out = applyThresh(out, self.threshD, self.thresh_mode)
		out = applyRandomZeros(out, self.zero_chance)
		self.resD = out
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		out = self.softmax(out)
		return out


def ResNet18(num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
	return ResNet(BasicBlock, [2, 2, 2, 2], num_channels, num_classes, thresh_arrs, zero_chance, thresh_mode)


def ResNet34(num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
	return ResNet(BasicBlock, [3, 4, 6, 3], num_channels, num_classes, thresh_arrs, zero_chance, thresh_mode)


def ResNet50(num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
	return ResNet(Bottleneck, [3, 4, 6, 3], num_channels, num_classes, thresh_arrs, zero_chance, thresh_mode)


def ResNet101(num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
	return ResNet(Bottleneck, [3, 4, 23, 3], num_channels, num_classes, thresh_arrs, zero_chance, thresh_mode)


def ResNet152(num_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
	return ResNet(Bottleneck, [3, 8, 36, 3], num_channels, num_classes, thresh_arrs, zero_chance, thresh_mode)




#---------------------------------------------------------------------------
#   VGG19 (Small Size Images)
#---------------------------------------------------------------------------


cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, in_channels=3, num_classes=10, thresh_arrs=None, zero_chance=0.0, thresh_mode='low'):
		super(VGG, self).__init__()
		self.features = self._make_layers(in_channels, cfg[vgg_name])
		self.classifier = nn.Linear(512, num_classes)

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

		# Keep track of the random zeros
		self.zero_chance = zero_chance

	def forward(self, x):

		x = self.features[0](x)   #(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[1](x)   #(1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[2](x)   #(2): ReLU(inplace=True)
		x = self.features[3](x)   #(3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[4](x)   #(4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[5](x)   #(5): ReLU(inplace=True)
		x = applyThresh(x, self.threshA, self.thresh_mode)
		x = applyRandomZeros(x, self.zero_chance)
		self.resA = x

		x = self.features[6](x)   #(6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		x = self.features[7](x)   #(7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[8](x)   #(8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[9](x)   #(9): ReLU(inplace=True)
		x = self.features[10](x)  #(10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[11](x)  #(11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[12](x)  #(12): ReLU(inplace=True)
		x = applyThresh(x, self.threshB, self.thresh_mode)
		x = applyRandomZeros(x, self.zero_chance)
		self.resB = x

		x = self.features[13](x)  #(13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		x = self.features[14](x)  #(14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[15](x)  #(15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[16](x)  #(16): ReLU(inplace=True)
		x = self.features[17](x)  #(17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[18](x)  #(18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[19](x)  #(19): ReLU(inplace=True)
		x = self.features[20](x)  #(20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[21](x)  #(21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[22](x)  #(22): ReLU(inplace=True)
		x = self.features[23](x)  #(23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[24](x)  #(24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[25](x)  #(25): ReLU(inplace=True)
		x = applyThresh(x, self.threshC, self.thresh_mode)
		x = applyRandomZeros(x, self.zero_chance)
		self.resC = x

		x = self.features[26](x)  #(26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		x = self.features[27](x)  #(27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[28](x)  #(28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[29](x)  #(29): ReLU(inplace=True)
		x = self.features[30](x)  #(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[31](x)  #(31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[32](x)  #(32): ReLU(inplace=True)
		x = self.features[33](x)  #(33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[34](x)  #(34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[35](x)  #(35): ReLU(inplace=True)
		x = self.features[36](x)  #(36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[37](x)  #(37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[38](x)  #(38): ReLU(inplace=True)
		x = applyThresh(x, self.threshD, self.thresh_mode)
		x = applyRandomZeros(x, self.zero_chance)
		self.resD = x

		x = self.features[39](x)  #(39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		x = self.features[40](x)  #(40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[41](x)  #(41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[42](x)  #(42): ReLU(inplace=True)
		x = self.features[43](x)  #(43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[44](x)  #(44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[45](x)  #(45): ReLU(inplace=True)
		x = self.features[46](x)  #(46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[47](x)  #(47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[48](x)  #(48): ReLU(inplace=True)
		x = self.features[49](x)  #(49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		x = self.features[50](x)  #(50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		x = self.features[51](x)  #(51): ReLU(inplace=True)
		x = applyThresh(x, self.threshE, self.thresh_mode)
		x = applyRandomZeros(x, self.zero_chance)
		self.resE = x

		x = self.features[52](x)  #(52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		out = self.features[53](x)  #(53): AvgPool2d(kernel_size=1, stride=1, padding=0)


		#out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		out = F.softmax(out)
		return out

	def _make_layers(self, in_channels, cfg):
		layers = []
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
					nn.BatchNorm2d(x),
					nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)



