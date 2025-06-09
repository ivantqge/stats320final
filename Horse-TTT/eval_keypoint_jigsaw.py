#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm

from horse_dataloader import get_horse_dataloaders, get_horse_domain_dataloaders
from keypoint_jigsaw_model import KeypointJigsawModel

class CombinedKeypointLoss(nn.Module):
    def __init__(self, loc_weight=0.05):
        super(CombinedKeypointLoss, self).__init__()
        self.loc_weight = loc_weight
        self.scoremap_loss = nn.BCEWithLogitsLoss()
        self.offset_loss = nn.SmoothL1Loss()
        
    def forward(self, pred_scoremaps, pred_offsets, target_heatmaps):
        # dimensions
        batch_size = pred_scoremaps.size(0)
        num_keypoints = pred_scoremaps.size(1)
        h, w = pred_scoremaps.size(2), pred_scoremaps.size(3)
        
        # cross-entropy loss
        scoremap_loss = self.scoremap_loss(pred_scoremaps, target_heatmaps)
        
        # flatten
        pred_flat = pred_scoremaps.view(batch_size, num_keypoints, -1)
        target_flat = target_heatmaps.view(batch_size, num_keypoints, -1)
        
        # indices w/ max probs
        _, pred_idx = torch.max(pred_flat, dim=2)  # [batch, num_keypoints]
        _, target_idx = torch.max(target_flat, dim=2)  # [batch, num_keypoints]
        
        # x, y coords
        scale_factor = 256.0 / 32.0  # 8.0
        pred_y = (pred_idx // w).float() * scale_factor
        pred_x = (pred_idx % w).float() * scale_factor
        target_y = (target_idx // w).float() * scale_factor
        target_x = (target_idx % w).float() * scale_factor
        
        # offsets from huber loss
        pred_y_idx = (pred_y / scale_factor).long().clamp(0, h-1)
        pred_x_idx = (pred_x / scale_factor).long().clamp(0, w-1)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_keypoints)
        keypoint_indices = torch.arange(num_keypoints).view(1, -1).expand(batch_size, -1)
        pred_offsets_at_points = pred_offsets[batch_indices, keypoint_indices, :, pred_y_idx, pred_x_idx]
        
        # target offsets
        target_offsets = torch.stack([
            target_x - pred_x,
            target_y - pred_y
        ], dim=2)  # [batch, num_keypoints, 2]
        
        # huber loss on offsets
        target_flat_orig = target_heatmaps.view(batch_size, num_keypoints, -1)
        valid_mask = (target_flat_orig.sum(dim=2) > 0).unsqueeze(-1)  # [batch, num_keypoints, 1]
        
        if valid_mask.sum() > 0:  # if valid kps
            valid_pred_offsets = pred_offsets_at_points[valid_mask.expand_as(pred_offsets_at_points)]
            valid_target_offsets = target_offsets[valid_mask.expand_as(target_offsets)]
            loc_loss = self.offset_loss(valid_pred_offsets, valid_target_offsets)
        else:
            loc_loss = torch.tensor(0.0, device=pred_offsets_at_points.device)
        
        # combine losses
        total_loss = scoremap_loss + self.loc_weight * loc_loss
        
        return total_loss, scoremap_loss, loc_loss

def extract_coordinates_from_heatmaps(scoremaps, offsets):
    """Extract keypoint coordinates from heatmaps with offset refinement"""
    batch_size, num_keypoints, h, w = scoremaps.shape
    
    # argmax locations from scoremaps
    pred_flat = scoremaps.view(batch_size, num_keypoints, -1)
    _, pred_idx = torch.max(pred_flat, dim=2)  # [batch, num_keypoints]
    
    # x, y coords
    scale_factor = 256.0 / 32.0  # changing heatmap size from 16 by 16 to 32 by 32
    pred_y = (pred_idx // w).float() * scale_factor
    pred_x = (pred_idx % w).float() * scale_factor
    
    # offsets from huber loss
    if offsets is not None:
        pred_y_idx = (pred_y / scale_factor).long().clamp(0, h-1)
        pred_x_idx = (pred_x / scale_factor).long().clamp(0, w-1)
        
        # batch indices for gathering offsets
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_keypoints)
        keypoint_indices = torch.arange(num_keypoints).view(1, -1).expand(batch_size, -1)
        
        # offsets at predicted locations
        pred_offsets_at_points = offsets[batch_indices, keypoint_indices, :, pred_y_idx, pred_x_idx]
        
        # apply offsets to get refined coordinates
        refined_x = pred_x + pred_offsets_at_points[:, :, 0]
        refined_y = pred_y + pred_offsets_at_points[:, :, 1]
    else:
        # no offset refinement for ground truth
        refined_x = pred_x
        refined_y = pred_y
    
    # new coordinates 
    coordinates = torch.stack([refined_x, refined_y], dim=2)
    
    return coordinates

def extract_coordinates_from_target_heatmaps(target_heatmaps):
    """Extract ground truth coordinates from target heatmaps"""
    batch_size, num_keypoints, h, w = target_heatmaps.shape
    
    # target kps
    target_flat = target_heatmaps.view(batch_size, num_keypoints, -1)
    _, target_idx = torch.max(target_flat, dim=2)  # [batch, num_keypoints]
    
    # x, y coords
    target_y = (target_idx // w).float() * scale_factor
    target_x = (target_idx % w).float() * scale_factor
    
    # new coordinates 
    coordinates = torch.stack([target_x, target_y], dim=2)
    
    return coordinates

def apply_quadrant_ssh(inputs):
    """Apply quadrant-based self-supervised task (from train_mouse.py)"""
    B, C, H, W = inputs.shape
    # even dimensions for splitting
    H_half = H // 2
    W_half = W // 2
    
    patches = []
    patch_labels = []
    
    for b in range(B):
        # quadrants with adaptive pooling to ensure consistent sizes
        q1 = inputs[b:b+1, :, :H_half, :W_half]
        q2 = inputs[b:b+1, :, :H_half, W_half:]
        q3 = inputs[b:b+1, :, H_half:, :W_half]
        q4 = inputs[b:b+1, :, H_half:, W_half:]
        
        # min height and width across quadrants
        min_H = min(q1.size(2), q2.size(2), q3.size(2), q4.size(2))
        min_W = min(q1.size(3), q2.size(3), q3.size(3), q4.size(3))
        
        # resize all quadrants to the minimum size
        q1 = nn.functional.adaptive_avg_pool2d(q1, (min_H, min_W))
        q2 = nn.functional.adaptive_avg_pool2d(q2, (min_H, min_W))
        q3 = nn.functional.adaptive_avg_pool2d(q3, (min_H, min_W))
        q4 = nn.functional.adaptive_avg_pool2d(q4, (min_H, min_W))
        
        quadrants = [q1, q2, q3, q4]
        order = torch.randperm(4)
        
        for idx in order:
            patches.append(quadrants[idx].squeeze(0))
            patch_labels.append(idx)
    
    patches = torch.stack(patches).to(inputs.device)
    patch_labels = torch.tensor(patch_labels).to(inputs.device)
    
    return patches, patch_labels

def apply_left_right_ssh(inputs):
    """Apply left/right-based self-supervised task"""
    B, C, H, W = inputs.shape
    
    patches = []
    patch_labels = []
    
    for b in range(B):
        # left and right halves
        left_half = inputs[b:b+1, :, :, :W//2]
        right_half = inputs[b:b+1, :, :, W//2:]
        
        # resize to ensure same dimensions
        min_W = min(left_half.size(3), right_half.size(3))
        left_half = nn.functional.adaptive_avg_pool2d(left_half, (H, min_W))
        right_half = nn.functional.adaptive_avg_pool2d(right_half, (H, min_W))
        
        halves = [left_half, right_half]
        order = torch.randperm(2)
        
        for idx in order:
            patches.append(halves[idx].squeeze(0))
            patch_labels.append(idx)
    
    patches = torch.stack(patches).to(inputs.device)
    patch_labels = torch.tensor(patch_labels).to(inputs.device)
    
    return patches, patch_labels

def compute_pck(pred_coords, gt_coords, threshold=0.1):
    """Simple PCK calculation with nose-to-eye normalization"""
    # pred_coords: (B, 22, 2), gt_coords: (B, 22, 2)
    
    # nose eye dist
    nose_coords = gt_coords[:, 0, :]  # (B, 2)
    eye_coords = gt_coords[:, 1, :]   # (B, 2)
    nose_eye_dist = torch.norm(nose_coords - eye_coords, dim=1)  # (B,)
    

    # valid_samples = (nose_eye_dist >= 0.0)  # in case no nose eye dist then messes up pck because it becomes 0
    
    # if not valid_samples.any():
    #     return 0.0, 0
    # If nose_eye_dist == 0, set it to some threshold distances
    nose_eye_dist_modified = torch.where(nose_eye_dist == 0, 
                                        torch.tensor(50.0, device=nose_eye_dist.device), 
                                        nose_eye_dist)
    
    # distances between pred and gt
    distances = torch.norm(pred_coords - gt_coords, dim=2)  # (B, 22)
    
    # don't include kps that don't exist
    valid_keypoints = ~((gt_coords[:, :, 0] == 0) & (gt_coords[:, :, 1] == 0))  # (B, 22)
    
    # threshold 
    threshold_distances = nose_eye_dist_modified.unsqueeze(1)  # (B, 1) -> (B, 22)
    correct = (distances < 0.3 * threshold_distances) & valid_keypoints  # (B, 22)
    total_valid = valid_keypoints.sum().item()
    total_correct = correct.sum().item()
    
    if total_valid == 0:
        return 0.0, 0
    
    pck = total_correct / total_valid
    return pck, total_valid

def evaluate_model(model, data_loader, criterion, device, ttt_steps=0, ttt_lr=0.001):
    """Evaluate model on data with both loss and PCK metrics, optionally with TTT"""
    model.eval()
    total_loss = 0
    total_scoremap_loss = 0
    total_loc_loss = 0
    num_batches = len(data_loader)
    
    # ttt tracking
    if ttt_steps > 0:
        print(f"Test-Time Training enabled: {ttt_steps} steps, LR: {ttt_lr}")
        total_loss_before_ttt = 0
        total_loss_after_ttt = 0
    
    # pck values
    total_valid_keypoints = 0
    thresholds = [0.3]
    pck_results = {th: [] for th in thresholds}
    
    ssh_criterion = nn.CrossEntropyLoss()

    print("Evaluating keypoint jigsaw model...")
    
    for batch in tqdm(data_loader, desc='Evaluating'):
        images = batch['image'].to(device)
        target_heatmaps = batch['heatmaps'].to(device)
        
        if ttt_steps > 0:

            model.train()  # enable training mode for ttt
            
            # freeze batchnorm parameters (idk why this makes it better)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            
            # optimizer for ttt (also try only optimize last backbone layer + jigsaw head)
            # ttt_optimizer = torch.optim.SGD(
            #     list(model.shared_backbone[7].parameters()) + list(model.jigsaw_head.parameters()), 
            #     lr=ttt_lr, momentum=0.0
            # )
            ttt_optimizer = torch.optim.Adam(
                list(model.shared_backbone.parameters()) + list(model.jigsaw_head.parameters()),
                lr=ttt_lr
            )
                        
            # TTT steps using self-supervised jigsaw task from 2016 paper
            for ttt_step in range(ttt_steps):
                ttt_optimizer.zero_grad()
                
                # jigsaw puzzles from test images (no labels needed)
                ssh_inputs, ssh_labels = apply_left_right_ssh(images)
                
                # forward + backward pass through jigsaw head
                ssh_pred = model.forward_jigsaw(ssh_inputs)
                
                ssh_loss = ssh_criterion(ssh_pred, ssh_labels)
                
                ssh_loss.backward()
                ttt_optimizer.step()
            
            model.eval()
            with torch.no_grad():
                pred_scoremaps, pred_offsets = model.forward_keypoints(images)
                loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
        else:
            # eval w/o TTT but on model trained w/ SSHead 
            model.eval()
            with torch.no_grad():
                pred_scoremaps, pred_offsets = model.forward_keypoints(images)
                loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
        
        total_loss += loss.item()
        total_scoremap_loss += scoremap_loss.item()
        total_loc_loss += loc_loss.item()
        
        # coordinates for pck
        pred_coords = extract_coordinates_from_heatmaps(pred_scoremaps, pred_offsets)
        gt_coords = extract_coordinates_from_heatmaps(target_heatmaps, None)
        
        # pck for different thresholds (just using 0.3 for now since that's what the paper uses)
        batch_cts = 0
        batch_scores = 0
        for threshold in thresholds:
            pck_score, valid_count = compute_pck(pred_coords, gt_coords, threshold)
            pck_results[threshold].append((pck_score, valid_count))
            batch_cts += valid_count
            batch_scores += pck_score * valid_count
        
        # if batch_cts != 0:
        #     print(f"Batch PCK: {batch_scores / batch_cts}")
       
    
    # averages
    avg_loss = total_loss / num_batches
    avg_scoremap_loss = total_scoremap_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    


    # overall pck scores
    overall_pck = {}
    for threshold in thresholds:
        total_correct = 0
        total_valid = 0
        for pck_score, valid_count in pck_results[threshold]:
            total_correct += pck_score * valid_count
            total_valid += valid_count
        
        if total_valid > 0:
            overall_pck[threshold] = total_correct / total_valid
        else:
            overall_pck[threshold] = 0.0
    
    return avg_loss, avg_scoremap_loss, avg_loc_loss, overall_pck, total_valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='final_keypoint_leftright_checkpoint_class2.pth', 
                       help='Path to keypoint jigsaw model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--domain', choices=['train', 'within', 'across', 'all'], default='train',
                       help='Which domain to evaluate on')
    parser.add_argument('--ttt_steps', type=int, default=0,
                       help='Number of test-time training steps (0 disables TTT)')
    parser.add_argument('--ttt_lr', type=float, default=0.001,
                       help='Learning rate for test-time training')
    args = parser.parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    split_file = os.path.join(root_dir, f'TrainTestInfo_shuffle{args.shuffle}.csv')
    
    # appropriate dataloader based on domain
    if args.domain == 'train':
        train_loader, _ = get_horse_dataloaders(
            root_dir=root_dir,
            split_file=split_file,
            batch_size=args.batch_size,
            shuffle=False
        )
        eval_loader = train_loader
        print(f"Evaluating on training set: {len(train_loader.dataset)} samples")
        
    elif args.domain in ['within', 'across']:
        within_domain_loader, across_domain_loader = get_horse_domain_dataloaders(
            root_dir=root_dir,
            split_file=split_file,
            batch_size=args.batch_size
        )
        
        if args.domain == 'within':
            eval_loader = within_domain_loader
            print(f"Evaluating on within-domain test: {len(within_domain_loader.dataset)} samples")
        else:
            eval_loader = across_domain_loader
            print(f"Evaluating on across-domain test: {len(across_domain_loader.dataset)} samples")
            
    else:  # 'all'
        _, test_loader = get_horse_dataloaders(
            root_dir=root_dir,
            split_file=split_file,
            batch_size=args.batch_size,
            shuffle=False
        )
        eval_loader = test_loader
        print(f"Evaluating on combined test set: {len(test_loader.dataset)} samples")
    
    # keypoint jigsaw model with diff architectures (tried to increase/reduce the complexity of each separate head)
    checkpoint_path = os.path.join(root_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = KeypointJigsawModel.load_from_checkpoint(checkpoint_path, device).to(device)
    
    # try loading models (if not epoch info then just loaded successfully)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'epoch' in checkpoint:
            print(f"Model loaded from epoch {checkpoint['epoch']}")
        else:
            print("Model loaded successfully")
    except:
        print("Model loaded successfully")
    
    # criterion (same as training)
    criterion = CombinedKeypointLoss(loc_weight=0.05)
    
    # evaluate
    avg_loss, avg_scoremap_loss, avg_loc_loss, pck_scores, total_valid = evaluate_model(
        model, eval_loader, criterion, device, args.ttt_steps, args.ttt_lr
    )
    
    print("\n" + "="*60)
    print(f"KEYPOINT JIGSAW MODEL EVALUATION RESULTS ({args.domain.upper()} SET)")
    print("="*60)
    print(f'Loss: {avg_loss:.4f}')
    print(f'  Scoremap Loss: {avg_scoremap_loss:.4f}')
    print(f'  Location Loss: {avg_loc_loss:.4f}')
    
    print()
    print("PCK Scores:")
    for threshold in sorted(pck_scores.keys()):
        print(f'  PCK@{threshold}: {pck_scores[threshold]:.3f}')
    print("="*60)

if __name__ == '__main__':
    main() 