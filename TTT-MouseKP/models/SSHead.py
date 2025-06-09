from torch import nn
import math
import copy
import torch.nn.functional as F

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def extractor_from_layer3(net):
	# for resnetcifar, no avgpool in ResNet build for mouse keypoint detection
	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu, net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	# works for both ResNetMouse/ResNetCifar 
	layers = [net.conv1, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
	"""
	For ResNetMouse 
	"""
	head_layers = []
	
	head_layers.extend([copy.deepcopy(net.layer3), copy.deepcopy(net.bn), copy.deepcopy(net.relu)])
	
	head_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
	
	# same layers
	head_layers.append(ViewFlatten())
	head_layers.append(nn.Linear(64 * width, classes))
	
	return nn.Sequential(*head_layers)
