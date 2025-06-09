import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from jigsaw_utils import JigsawHead, JigsawPuzzleGenerator

class KeypointJigsawModel(nn.Module):
    """
    Same as horse_model.py but with jigsaw head
    """
    def __init__(self, num_keypoints=22, jigsaw_classes=100, use_legacy_jigsaw=False, use_legacy_keypoint=False):
        super(KeypointJigsawModel, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        self.shared_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.backbone_channels = 2048  # resnet50 layer4 output channels
        
        # keypoint detection head
        self.keypoint_head = self._make_keypoint_head(num_keypoints, use_legacy_keypoint)
        
        # additional upsampling to ensure 32 by 32 output
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        
        # jigsaw puzzle head (attached to shared backbone)
        self.jigsaw_head = JigsawHead(
            input_channels=self.backbone_channels,
            num_classes=jigsaw_classes,
            use_legacy_arch=use_legacy_jigsaw
        )
        
        # jigsaw puzzle generator
        self.jigsaw_generator = JigsawPuzzleGenerator(
            grid_size=3, 
            num_permutations=jigsaw_classes
        )
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cpu'):
        """Load model from checkpoint with automatic architecture detection"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # detect keypoint head architecture (legacy has more layers)
        keypoint_keys = [k for k in state_dict.keys() if 'keypoint_head' in k]
        max_keypoint_layer = 0
        for key in keypoint_keys:
            if 'keypoint_head.' in key:
                layer_num = int(key.split('.')[1])
                max_keypoint_layer = max(max_keypoint_layer, layer_num)
        
        use_legacy_keypoint = max_keypoint_layer >= 9
        
        # detect number of keypoints from keypoint head
        num_keypoints = None
        # Find the final output layer
        if use_legacy_keypoint:
            final_key = 'keypoint_head.9.weight' 
        else:
            final_key = 'keypoint_head.6.weight' 
            
        if final_key in state_dict:
            num_keypoints = state_dict[final_key].shape[0] // 3  
        
        if num_keypoints is None:
            print("Warning: Could not detect number of keypoints, defaulting to 22")
            num_keypoints = 22
        
        # one file to be compatible with both
        jigsaw_classes = None
        if 'jigsaw_head.classifier.6.weight' in state_dict:
            jigsaw_classes = state_dict['jigsaw_head.classifier.6.weight'].shape[0]
        elif 'jigsaw_head.classifier.6.bias' in state_dict:
            jigsaw_classes = state_dict['jigsaw_head.classifier.6.bias'].shape[0]
        elif 'jigsaw_head.classifier.weight' in state_dict:
            jigsaw_classes = state_dict['jigsaw_head.classifier.weight'].shape[0]
        elif 'jigsaw_head.classifier.bias' in state_dict:
            jigsaw_classes = state_dict['jigsaw_head.classifier.bias'].shape[0]
        
        if jigsaw_classes is None:
            print("Warning: Could not detect number of jigsaw classes, defaulting to 9")
            jigsaw_classes = 9
        
        jigsaw_keys = [k for k in state_dict.keys() if 'jigsaw_head' in k]
        has_conv_layers = any('conv_layers' in key for key in jigsaw_keys)
        has_legacy_classifier = any('classifier.0.weight' in key for key in jigsaw_keys)
        use_legacy_jigsaw = has_legacy_classifier and not has_conv_layers
        
        print(f"Detected: {num_keypoints} keypoints, {jigsaw_classes} jigsaw classes")
        print(f"Using {'legacy' if use_legacy_keypoint else 'new'} keypoint architecture")
        print(f"Using {'legacy' if use_legacy_jigsaw else 'new'} jigsaw architecture")
        
        #  create model for appropriate architecture (mainly for inference)
        model = cls(
            num_keypoints=num_keypoints,
            jigsaw_classes=jigsaw_classes,
            use_legacy_jigsaw=use_legacy_jigsaw,
            use_legacy_keypoint=use_legacy_keypoint
        )
        
        model.load_state_dict(state_dict)
        
        return model
    
    def _make_keypoint_head(self, num_keypoints, use_legacy_keypoint=False):
        """Create keypoint detection head with deconvolution layers"""
        if use_legacy_keypoint: # a lot more layers, maybe too much but realized that the SSHead was having severe impacts on keypoint detection
            return nn.Sequential(
                # deconv layer 1: 2048 -> 256, upsample 2x
                nn.ConvTranspose2d(self.backbone_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # output scoremaps and also offsets for huber loss
                nn.Conv2d(256, num_keypoints + num_keypoints * 2, kernel_size=1)
            )
        else:
            # less layers
            return nn.Sequential(
                # deconv layer 1: 2048 -> 256, upsample 2x
                nn.ConvTranspose2d(self.backbone_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                 
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # output scoremaps and also offets for huber loss
                nn.Conv2d(256, num_keypoints + num_keypoints * 2, kernel_size=1)
            )
    
    def forward_keypoints(self, x):
        """Forward pass for keypoint detection"""
        features = self.shared_backbone(x)
        
        keypoint_output = self.keypoint_head(features)
        
        keypoint_output = self.upsample(keypoint_output)
        
        num_keypoints = keypoint_output.shape[1] // 3
        scoremaps = keypoint_output[:, :num_keypoints]
        offset_maps = keypoint_output[:, num_keypoints:].view(
            keypoint_output.shape[0], num_keypoints, 2, 
            keypoint_output.shape[2], keypoint_output.shape[3]
        )
        
        return scoremaps, offset_maps
    
    def forward_jigsaw(self, x):
        """Forward pass for jigsaw puzzle prediction"""
        
        # shared feature extraction  
        features = self.shared_backbone(x)
        
        jigsaw_output = self.jigsaw_head(features)
        
        return jigsaw_output
    
    def forward(self, x, task='keypoints'):
        """Forward pass - can specify which task to run"""
        if task == 'keypoints':
            return self.forward_keypoints(x)
        elif task == 'jigsaw':
            return self.forward_jigsaw(x)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def create_jigsaw_batch(self, images):
        """Create jigsaw puzzles from input images"""
        return self.jigsaw_generator.create_jigsaw_batch(images)