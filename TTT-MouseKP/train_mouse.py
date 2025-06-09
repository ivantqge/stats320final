from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from models.ResNet import ResNetMouse
from models.SSHead import extractor_from_layer2, head_on_layer2, ExtractorHead

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mouse')
parser.add_argument('--dataroot', default='./data/mouse_dataset/')
parser.add_argument('--train_frames_dir', default='train/images')
parser.add_argument('--train_masks_dir', default='train/masks')
parser.add_argument('--val_frames_dir', default='val/images')
parser.add_argument('--val_masks_dir', default='val/masks')
parser.add_argument('--shared', default='layer2')
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--group_norm', default=8, type=int)
parser.add_argument('--num_keypoints', default=5, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
########################################################################
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--nepoch', default=50, type=int)
parser.add_argument('--milestone_1', default=30, type=int)
parser.add_argument('--milestone_2', default=60, type=int)
########################################################################
parser.add_argument('--disable_ssh', action='store_true', help='Disable SSH loss for training')
parser.add_argument('--ssh_task', default='quadrant', choices=['rotation', 'quadrant', 'scale'], 
                   help='Self-supervised task type')
parser.add_argument('--ssh_weight', default=0.1, type=float, help='Weight for SSH loss')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], 
                   help='Optimizer type')
parser.add_argument('--outf', default='./results/mouse_tracking')

args = parser.parse_args()

# Adjust output directory based on configuration
if args.disable_ssh:
    args.outf = './results/mouse_tracking_no_ssh'

my_makedir(args.outf)

# Build model
print("Using custom ResNet architecture")
net = ResNetMouse(args.depth, args.width, num_keypoints=args.num_keypoints, kernel_size=args.kernel_size).cuda()

# Create SSH components if not disabled
if not args.disable_ssh:
    ext = extractor_from_layer2(net)
    head = head_on_layer2(net, args.width, 4)
    ssh = ExtractorHead(ext, head).cuda()
else:
    ssh = None
    head = None

# Setup optimizer
if args.disable_ssh:
    optimizer_params = net.parameters()
else:
    optimizer_params = list(net.parameters()) + list(head.parameters())

# Create optimizer
if args.optimizer == 'sgd':
    optimizer = optim.SGD(optimizer_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.Adam(optimizer_params, lr=args.lr, weight_decay=1e-4)

# Prepare data
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

# Learning rate scheduler 
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.2)

# Loss functions - updated for heatmap training
criterion_heatmap = nn.BCEWithLogitsLoss().cuda()  # Bernoulli loss for heatmaps
if not args.disable_ssh:
    criterion_ssh = nn.CrossEntropyLoss().cuda()

def apply_quadrant_ssh(inputs):
    """Apply quadrant-based self-supervised task"""
    B, C, H, W = inputs.shape

    H_half = H // 2
    W_half = W // 2
    
    patches = []
    patch_labels = []
    
    for b in range(B):

        # 4 quadrants
        q1 = inputs[b:b+1, :, :H_half, :W_half]
        q2 = inputs[b:b+1, :, :H_half, W_half:]
        q3 = inputs[b:b+1, :, H_half:, :W_half]
        q4 = inputs[b:b+1, :, H_half:, W_half:]
        
        min_H = min(q1.size(2), q2.size(2), q3.size(2), q4.size(2))
        min_W = min(q1.size(3), q2.size(3), q3.size(3), q4.size(3))
        
        # resizing if diff sizes
        q1 = nn.functional.adaptive_avg_pool2d(q1, (min_H, min_W))
        q2 = nn.functional.adaptive_avg_pool2d(q2, (min_H, min_W))
        q3 = nn.functional.adaptive_avg_pool2d(q3, (min_H, min_W))
        q4 = nn.functional.adaptive_avg_pool2d(q4, (min_H, min_W))
        
        quadrants = [q1, q2, q3, q4]
        order = torch.randperm(4)
        
        for idx in order:
            patches.append(quadrants[idx].squeeze(0))
            patch_labels.append(idx)
    
    patches = torch.stack(patches).cuda()
    patch_labels = torch.tensor(patch_labels).cuda()
    
    return patches, patch_labels

all_losses = []
print('Training...')
print(f'Model: Custom ResNet')
print(f'SSH Task: {args.ssh_task}, SSH Weight: {args.ssh_weight}, Optimizer: {args.optimizer}')

if args.disable_ssh:
    print('Epoch\tHeatmap Loss\tLR')
else:
    print('Epoch\tHeatmap Loss\tSSH Loss\tTotal Loss\tLR')

for epoch in range(1, args.nepoch + 1):
    net.train()
    if ssh is not None:
        ssh.train()
    epoch_losses = []
    epoch_ssh_losses = []
    epoch_total_losses = []

    for batch_idx, (inputs, targets) in enumerate(trloader):
        optimizer.zero_grad()
        
        # heatmap prediction
        inputs_heatmap, targets_heatmap = inputs.cuda(), targets.cuda()
        outputs_heatmap = net(inputs_heatmap)  # Shape: [B, num_keypoints, H, W]
        loss_heatmap = criterion_heatmap(outputs_heatmap, targets_heatmap)
        
        total_loss = loss_heatmap
        ssh_loss_value = 0.0
        
        if not args.disable_ssh:

            # Self-supervised task - quadrant prediction
            inputs_ssh, labels_ssh = apply_quadrant_ssh(inputs)
            
            outputs_ssh = ssh(inputs_ssh)
            loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
            ssh_loss_value = loss_ssh.item()
            
            # Combined loss with configurable weighting
            total_loss = loss_heatmap + args.ssh_weight * loss_ssh
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if ssh is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        
        optimizer.step()
        
        # logs
        epoch_losses.append(loss_heatmap.item())
        epoch_ssh_losses.append(ssh_loss_value)
        epoch_total_losses.append(total_loss.item())
    
    # calculate average losses
    avg_heatmap_loss = np.mean(epoch_losses)
    avg_ssh_loss = np.mean(epoch_ssh_losses)
    avg_total_loss = np.mean(epoch_total_losses)
    
    if args.disable_ssh:
        all_losses.append(avg_heatmap_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}/{args.nepoch}\t{avg_heatmap_loss:.4f}\t{current_lr:.6f}')
    else:
        all_losses.append([avg_heatmap_loss, avg_ssh_loss, avg_total_loss])
        current_lr = optimizer.param_groups[0]['lr']
        print(f'{epoch}/{args.nepoch}\t{avg_heatmap_loss:.4f}\t{avg_ssh_loss:.4f}\t{avg_total_loss:.4f}\t{current_lr:.6f}')
    
    scheduler.step()

# Save final model
if args.disable_ssh:
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': all_losses
    }
else:
    state = {
        'net': net.state_dict(),
        'head': head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': all_losses
    }
torch.save(state, f'{args.outf}/model_final.pth') 