import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import permutations

class JigsawPuzzleGenerator:
    def __init__(self, grid_size=2, num_permutations=100):

        """
        grid_size: 3 means 3x3 grid (9 patches); can also do 2, etc. 
        num_permutations: number of predefined permutations to use
        """

        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        
        self.permutations = self._generate_permutations(num_permutations)
        self.num_classes = len(self.permutations)
        
    def _generate_permutations(self, num_permutations):

        perms = [list(range(self.num_patches))]
        
        # random permutations
        while len(perms) < num_permutations:
            perm = list(range(self.num_patches))
            random.shuffle(perm)
            if perm not in perms:
                perms.append(perm)
                
        return perms
    
    def create_jigsaw_batch(self, images):

        """
        Create jigsaw puzzles from a batch of images
        Args:
            images: (B, C, H, W) tensor
        Returns:
            jigsaw_images: (B, C, H, W) shuffled images
            labels: (B,) permutation indices
        """

        B, C, H, W = images.shape
        patch_h = H // self.grid_size
        patch_w = W // self.grid_size
        
        jigsaw_images = []
        labels = []
        
        for b in range(B):
            # patches
            patches = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    patch = images[b, :, 
                                 i*patch_h:(i+1)*patch_h, 
                                 j*patch_w:(j+1)*patch_w]
                    patches.append(patch)
            
            # random permutation
            perm_idx = random.randint(0, self.num_classes - 1)
            permutation = self.permutations[perm_idx]
            
            # shuffle patches according to permutation
            shuffled_patches = [patches[permutation[i]] for i in range(self.num_patches)]
            
            # reconstruct image
            jigsaw_img = torch.zeros_like(images[b])
            for idx, patch in enumerate(shuffled_patches):
                i = idx // self.grid_size
                j = idx % self.grid_size
                jigsaw_img[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patch
            
            jigsaw_images.append(jigsaw_img)
            labels.append(perm_idx)
        
        return torch.stack(jigsaw_images), torch.tensor(labels)

class JigsawHead(nn.Module):
    """Head for predicting spatial tasks (left/right, quadrants, etc.)"""
    def __init__(self, input_channels, num_classes, hidden_dim=256, use_legacy_arch=False):
        super(JigsawHead, self).__init__()
        
        self.use_legacy_arch = use_legacy_arch
        
        if use_legacy_arch:
            # legacy architecture (3-layer MLP) for backwards compatibility
            self.classifier = nn.Sequential(
                nn.Linear(input_channels, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # new arch with conv layers
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            # conv layers
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(inplace=True)
            )
            
            # final linear classifier
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if self.use_legacy_arch:
            # legacy forward pass
            # x: (B, C, H, W) features from shared backbone
            x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
            x = x.view(x.size(0), -1)  # (B, C) - flatten
            x = self.classifier(x)  # (B, num_classes)
            return x
        else:
            # new forward pass
            # x: (B, C, H, W) features from shared backbone
            x = self.global_pool(x)  # (B, C, 1, 1)
            x = self.conv_layers(x)  # (B, hidden_dim, 1, 1)
            x = x.view(x.size(0), -1)  # (B, hidden_dim) - flatten for linear layer
            x = self.classifier(x)  # (B, num_classes)
            return x
    
    @classmethod        
    def create_from_checkpoint(cls, input_channels, num_classes, checkpoint_state_dict, hidden_dim=256):

        # check if checkpoint contains legacy architecture
        jigsaw_keys = [k for k in checkpoint_state_dict.keys() if 'jigsaw_head' in k]
        has_conv_layers = any('conv_layers' in key for key in jigsaw_keys)
        has_legacy_classifier = any('classifier.0.weight' in key for key in jigsaw_keys)
        
        if has_legacy_classifier and not has_conv_layers:
            print("Detected legacy JigsawHead architecture - using backwards compatibility mode")
            return cls(input_channels, num_classes, hidden_dim, use_legacy_arch=True)
        else:
            print("Detected new JigsawHead architecture")
            return cls(input_channels, num_classes, hidden_dim, use_legacy_arch=False)