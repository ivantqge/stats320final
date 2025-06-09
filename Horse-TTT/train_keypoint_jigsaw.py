import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from keypoint_jigsaw_model import KeypointJigsawModel
import numpy as np

class CombinedKeypointLoss(nn.Module):
    def __init__(self, loc_weight=0.05):
        super(CombinedKeypointLoss, self).__init__()
        self.loc_weight = loc_weight
        self.scoremap_loss = nn.BCEWithLogitsLoss()
        self.offset_loss = nn.SmoothL1Loss()
        
    def forward(self, pred_scoremaps, pred_offsets, target_heatmaps):
        # batch size and dimensions
        batch_size = pred_scoremaps.size(0)
        num_keypoints = pred_scoremaps.size(1)
        h, w = pred_scoremaps.size(2), pred_scoremaps.size(3)
        
        # scoremap loss (bce)
        scoremap_loss = self.scoremap_loss(pred_scoremaps, target_heatmaps)
        
        # predicted and target keypoint locations
        pred_flat = pred_scoremaps.view(batch_size, num_keypoints, -1)
        target_flat = target_heatmaps.view(batch_size, num_keypoints, -1)
        
        # indices of max values
        _, pred_idx = torch.max(pred_flat, dim=2)  # [batch, num_keypoints]
        _, target_idx = torch.max(target_flat, dim=2)  # [batch, num_keypoints]
        
        # convert indices to x,y coordinates (pixel space, scaled to 256x256)
        scale_factor = 256.0 / 32.0  # 8.0
        pred_y = (pred_idx // w).float() * scale_factor
        pred_x = (pred_idx % w).float() * scale_factor
        target_y = (target_idx // w).float() * scale_factor
        target_x = (target_idx % w).float() * scale_factor
        
        # predicted offsets at the predicted locations
        # convert pixel coordinates back to heatmap indices for offset lookup
        pred_y_idx = (pred_y / scale_factor).long().clamp(0, h-1)
        pred_x_idx = (pred_x / scale_factor).long().clamp(0, w-1)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_keypoints)
        keypoint_indices = torch.arange(num_keypoints).view(1, -1).expand(batch_size, -1)
        pred_offsets_at_points = pred_offsets[batch_indices, keypoint_indices, :, pred_y_idx, pred_x_idx]
        
        # target offsets (difference between target and predicted locations)
        target_offsets = torch.stack([
            target_x - pred_x,
            target_y - pred_y
        ], dim=2)  # [batch, num_keypoints, 2]
        
        # only calculate loss for valid (non-NaN) keypoints
        target_flat_orig = target_heatmaps.view(batch_size, num_keypoints, -1)
        valid_mask = (target_flat_orig.sum(dim=2) > 0).unsqueeze(-1)  # [batch, num_keypoints, 1]
        
        if valid_mask.sum() > 0:  
            valid_pred_offsets = pred_offsets_at_points[valid_mask.expand_as(pred_offsets_at_points)]
            valid_target_offsets = target_offsets[valid_mask.expand_as(target_offsets)]
            loc_loss = self.offset_loss(valid_pred_offsets, valid_target_offsets)
        else:
            loc_loss = torch.tensor(0.0, device=pred_offsets_at_points.device)
        
        # combine losses    
        total_loss = scoremap_loss + self.loc_weight * loc_loss
        
        return total_loss, scoremap_loss, loc_loss

def apply_left_right_ssh(inputs):
    """Apply left/right-based self-supervised task (simplified from quadrants)"""
    B, C, H, W = inputs.shape
    # even dimensions for splitting
    W_half = W // 2
    
    patches = []
    patch_labels = []
    
    for b in range(B):
        # split into left and right halves
        left_half = inputs[b:b+1, :, :, :W_half]
        right_half = inputs[b:b+1, :, :, W_half:]
        
        # both halves are the same size
        min_H = min(left_half.size(2), right_half.size(2))
        min_W = min(left_half.size(3), right_half.size(3))
        
        # resize both halves to the minimum size
        left_half = nn.functional.adaptive_avg_pool2d(left_half, (min_H, min_W))
        right_half = nn.functional.adaptive_avg_pool2d(right_half, (min_H, min_W))
        
        halves = [left_half, right_half]
        order = torch.randperm(2)  # only 2 classes instead of 4
        
        for idx in order:
            patches.append(halves[idx].squeeze(0))
            patch_labels.append(idx)
    
    patches = torch.stack(patches).to(inputs.device)
    patch_labels = torch.tensor(patch_labels).to(inputs.device)
    
    return patches, patch_labels

def train_joint_model(model, criterion, train_loader, num_epochs=100, device='cuda', ssh_weight=0.1):
    """
    train model jointly on keypoint detection + left/right SSH task (also tried quadrants)
    """
    model = model.to(device)
    
    # optimizer for all parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    ssh_criterion = nn.CrossEntropyLoss()
    
    print(f'Training with SSH weight: {ssh_weight}')
    print('Epoch\tKeypoint Loss\tSSH Loss\tTotal Loss\tLR')
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_keypoint_losses = []
        epoch_ssh_losses = []
        epoch_total_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            
            # keypoint detection
            pred_scoremaps, pred_offsets = model.forward_keypoints(images)
            keypoint_loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
            
            # left/right prediction, also did quadrant, and can also do jigsaw (3x3, etc.)
            ssh_inputs, ssh_labels = apply_left_right_ssh(images)
            ssh_pred = model.forward_jigsaw(ssh_inputs)
            ssh_loss = ssh_criterion(ssh_pred, ssh_labels)
            
            total_loss = keypoint_loss + ssh_weight * ssh_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # logging
            epoch_keypoint_losses.append(keypoint_loss.item())
            epoch_ssh_losses.append(ssh_loss.item())
            epoch_total_losses.append(total_loss.item())
        
        scheduler.step()
        
        avg_keypoint_loss = np.mean(epoch_keypoint_losses)
        avg_ssh_loss = np.mean(epoch_ssh_losses)
        avg_total_loss = np.mean(epoch_total_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        #print(f'{epoch+1}/{num_epochs}\t{avg_keypoint_loss:.4f}\t{avg_ssh_loss:.4f}\t{avg_total_loss:.4f}\t{current_lr:.6f}')
        
        # save checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'keypoint_loss': avg_keypoint_loss,
                'ssh_loss': avg_ssh_loss,
            }, f'checkpoint_epoch_{epoch}.pth')
    
    final_checkpoint_path = f'final_keypoint_leftright_checkpoint_class2.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'keypoint_loss': avg_keypoint_loss,
        'ssh_loss': avg_ssh_loss,
        'ssh_weight': ssh_weight,
    }, final_checkpoint_path)
    print(f'Training completed! Final checkpoint saved to {final_checkpoint_path}')