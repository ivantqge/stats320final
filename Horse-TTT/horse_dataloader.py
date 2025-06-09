import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import h5py

def generate_heatmap(x, y, height, width, sigma=1):

    """Generate a single heatmap for a keypoint"""

    if np.isnan(x) or np.isnan(y):
        return np.zeros((height, width), dtype=np.float32)
    
    x = int(x * width)
    y = int(y * height)
    
    # meshgrid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    # trying gaussians at each keypoint
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    
    return gaussian

class HorseKeypointDataset(Dataset):
    def __init__(self, root_dir, split_file=None, transform=None, train=True, input_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            split_file (string): Path to the split file (train/test info)
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): If True, load training data, else load test data
            input_size (tuple): Size of the input images and heatmaps (height, width)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.input_size = input_size
        
        # loading from csv now, initially loaded for h5py and the keypoints were messed up ?
        csv_path = os.path.join(root_dir, 'training-datasets', 'iteration-0', 'UnaugmentedDataSet_HorsesMay8', 'CollectedData_Byron.csv')
        self.data = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        
        # split info
        if split_file:
            splits = pd.read_csv(split_file)
            if train:
                self.image_paths = splits['trainIndices'].dropna().tolist()
            else:
                within_domain = splits['testIndices_withinDomain'].dropna().tolist()
                across_domain = splits['testIndices_acrossDomain'].dropna().tolist()
                self.image_paths = within_domain + across_domain
        else:
            self.image_paths = [os.path.join('labeled-data', p) for p in self.data.index]
        
        # setting heatmap size to 32 by 32  
        self.heatmap_size = (32, 32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # load image
        img_full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_full_path).convert('RGB')
        orig_size = image.size  # (width, height)
        
        raw_keypoints = self.data.loc[img_path].values
        # print(img_path)
        # print(raw_keypoints)
        # print(orig_size)
        keypoints = raw_keypoints.reshape(-1, 2)  # shape: (22, 2) for x,y coordinates
        
        keypoints[:, 0] = keypoints[:, 0] * self.input_size[1] / orig_size[0]  # x coordinates
        keypoints[:, 1] = keypoints[:, 1] * self.input_size[0] / orig_size[1]  # y coordinates
        
        # heatmaps at each point
        heatmaps = np.zeros((len(keypoints), self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y):
                heatmap_x = x / self.input_size[1]  # normalize to [0,1]
                heatmap_y = y / self.input_size[0]  # normalize to [0,1]
                heatmaps[i] = generate_heatmap(heatmap_x, heatmap_y, self.heatmap_size[0], self.heatmap_size[1], sigma=0.5)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'heatmaps': torch.FloatTensor(heatmaps),
            'keypoints': torch.FloatTensor(keypoints),
            'path': img_path
        }

def get_horse_dataloaders(root_dir, split_file, batch_size=32, shuffle=True):
    """
    Create train and test dataloaders
    """
    input_size = (256, 256)  # using 256 by 256 here, slightly higher res b/c images are bigger but maybe should've used 224 by 224
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HorseKeypointDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=transform,
        train=True
    )
    
    test_dataset = HorseKeypointDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=transform,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader 

def get_horse_domain_dataloaders(root_dir, split_file, batch_size=32):

    """
    Create separate within-domain and across-domain test dataloaders
    """

    input_size = (256, 256)
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # split info from csv
    splits = pd.read_csv(split_file)
    
    # domain-specific image paths
    within_domain_paths = splits['testIndices_withinDomain'].dropna().tolist()
    across_domain_paths = splits['testIndices_acrossDomain'].dropna().tolist()
    
    # custom dataset class for specific image paths
    class DomainSpecificDataset(HorseKeypointDataset):
        def __init__(self, root_dir, image_paths, transform=None):
            super().__init__(root_dir, None, transform, False)
            self.image_paths = image_paths
    
    within_domain_dataset = DomainSpecificDataset(root_dir, within_domain_paths, transform)
    across_domain_dataset = DomainSpecificDataset(root_dir, across_domain_paths, transform)
    
    # dataloaders
    within_domain_loader = DataLoader(
        within_domain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    across_domain_loader = DataLoader(
        across_domain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return within_domain_loader, across_domain_loader 