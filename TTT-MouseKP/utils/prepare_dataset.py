import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(*NORM)])
mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
										transforms.ToTensor(),
										transforms.Normalize((0.1307,), (0.3081,))])

# Mouse dataset transforms (no random crop/flip for precise keypoint detection)
mouse_te_transforms = transforms.Compose([transforms.ToTensor(),
										 transforms.Normalize(*NORM)])
mouse_tr_transforms = transforms.Compose([transforms.ToTensor(),
										 transforms.Normalize(*NORM)])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

# new mousedataset to import images + masks
class MouseDataset(torch.utils.data.Dataset):
	def __init__(self, root, split='train', transform=None, num_keypoints=5):
		"""
		Mouse dataset for keypoint detection
		Args:
			root: path to mouse_data folder
			split: 'train' or 'val'
			transform: transforms to apply to images
			num_keypoints: number of keypoints (should be 5)
		"""
		self.root = root
		self.split = split
		self.transform = transform
		self.num_keypoints = num_keypoints
		
		self.images_dir = os.path.join(root, split, 'images')
		self.masks_dir = os.path.join(root, split, 'masks')
		
		self.image_files = [f for f in os.listdir(self.images_dir) 
						   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
		self.image_files.sort()
		
		print(f"Found {len(self.image_files)} images in {self.images_dir}")
	
	def __len__(self):
		return len(self.image_files)
	
	def __getitem__(self, idx):

		# koad image
		img_name = self.image_files[idx]
		img_path = os.path.join(self.images_dir, img_name)
		image = Image.open(img_path).convert('RGB')
		
		img_width, img_height = image.size
		
		# loading masks
		img_base = os.path.splitext(img_name)[0] 
		heatmaps = []
		
		for kp_idx in range(self.num_keypoints):
			# png id + kp + idx + .png
			mask_name = f"{img_base}_kp{kp_idx}.png"
			mask_path = os.path.join(self.masks_dir, mask_name)
			
			if not os.path.exists(mask_path):
				# if no mask
				print(f"Warning: No mask found for {mask_name}")
				heatmap = np.zeros((img_height, img_width), dtype=np.float32)
			else:
				# loading mask
				mask = Image.open(mask_path).convert('L') 
				mask = mask.resize((img_width, img_height)) 
				heatmap = np.array(mask, dtype=np.float32) / 255.0  
			
			heatmaps.append(heatmap)
		
		# stack heatmaps for each image
		heatmaps = np.stack(heatmaps, axis=0)
		heatmaps = torch.from_numpy(heatmaps).float()
		
		if self.transform:
			image = self.transform(image)
		
		return image, heatmaps

# ported over from previous code, just added a new args.dataset for mouse
def prepare_test_data(args):
	if args.dataset == 'cifar10':
		tesize = 10000
		if not hasattr(args, 'corruption') or args.corruption == 'original':
			print('Test on the original test set')
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
		elif args.corruption in common_corruptions:
			print('Test on %s level %d' %(args.corruption, args.level))
			teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
			teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
			teset.data = teset_raw

		elif args.corruption == 'cifar_new':
			from utils.cifar_new import CIFAR_New
			print('Test on CIFAR-10.1')
			teset = CIFAR_New(root=args.dataroot + 'CIFAR-10.1/datasets/', transform=te_transforms)
			permute = False
		else:
			raise Exception('Corruption not found!')
	elif args.dataset == 'mouse':
		print('Loading mouse validation data...')
		mouse_data_path = os.path.join(args.dataroot, 'mouse_dataset')
		teset = MouseDataset(mouse_data_path, split='val', transform=mouse_te_transforms, 
							num_keypoints=getattr(args, 'num_keypoints', 5))
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.workers)
	return teset, teloader

def prepare_train_data(args):
	print('Preparing data...')
	if args.dataset == 'cifar10':
		trset = torchvision.datasets.CIFAR10(root=args.dataroot,
										train=True, download=True, transform=tr_transforms)
	elif args.dataset == 'mouse':
		print('Loading mouse training data...')
		mouse_data_path = os.path.join(args.dataroot, 'mouse_dataset')
		trset = MouseDataset(mouse_data_path, split='train', transform=mouse_tr_transforms,
							num_keypoints=getattr(args, 'num_keypoints', 5))
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.workers)
	return trset, trloader
	