import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class KeypointResNet50(nn.Module):
    def __init__(self, num_keypoints=22):
        super(KeypointResNet50, self).__init__()
        
        # using pre-trained resnet
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # deconv layers to upsample to 32 by 32
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # additional upsampling to ensure 32 by 32 output
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        
        # scoremaps + offset 
        self.scoremap_head = nn.Conv2d(256, num_keypoints, kernel_size=1)
        self.offset_head = nn.Conv2d(256, num_keypoints * 2, kernel_size=1)  # x,y offsets for each keypoint
        
    def forward(self, x):
        input_size = x.size()[-2:]
        
        features = self.features(x)  
        
        x = self.deconv1(features)  
        x = self.upsample(x) 
        
        scoremaps = self.scoremap_head(x)  # [batch, num_keypoints, 32, 32]
        offsets = self.offset_head(x)  # [batch, num_keypoints*2, 32, 32]
        
        offsets = offsets.view(offsets.size(0), -1, 2, offsets.size(2), offsets.size(3))  # [batch, num_keypoints, 2, H/8, W/8]
        
        return scoremaps, offsets
    
    def get_keypoint_coordinates(self, heatmaps):
        """Extract keypoint coordinates from heatmaps using argmax"""
        batch_size = heatmaps.size(0)
        
        # probability distributions
        heatmaps_flat = heatmaps.view(batch_size, self.num_keypoints, -1)
        heatmaps_flat = F.softmax(heatmaps_flat, dim=2)
        
        # max locations
        max_vals, max_idx = torch.max(heatmaps_flat, dim=2)
        
        h, w = heatmaps.size(2), heatmaps.size(3)
        y_coords = (max_idx // w).float() / (h - 1)
        x_coords = (max_idx % w).float() / (w - 1)  
        
        # coords that are normalized
        coords = torch.stack([x_coords, y_coords, max_vals], dim=2)
        
        return coords 

def get_model(num_keypoints):
    """
    Create a KeypointResNet50 model
    
    Args:
        num_keypoints (int): Number of keypoints to detect
    """
    model = KeypointResNet50(num_keypoints)
    return model


def get_optimizer(model, lr=0.001, momentum=0.9, weight_decay=1e-4):
    """
    Create an optimizer for the model
    
    Args:
        model: The model to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay factor
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    return optimizer 