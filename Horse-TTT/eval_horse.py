import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from horse_dataloader import get_horse_dataloaders
from horse_dataloader import get_horse_domain_dataloaders

from horse_model import KeypointResNet50

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
        
        # cross-entropy loss (BCE)
        scoremap_loss = self.scoremap_loss(pred_scoremaps, target_heatmaps)
        
        # flatten 
        pred_flat = pred_scoremaps.view(batch_size, num_keypoints, -1)
        target_flat = target_heatmaps.view(batch_size, num_keypoints, -1)
        
        # indices w/ max probs
        _, pred_idx = torch.max(pred_flat, dim=2)  # [batch, num_keypoints]
        _, target_idx = torch.max(target_flat, dim=2)  # [batch, num_keypoints]
        
        # x, y coords
        scale_factor = 256.0 / 32.0  
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
        
        target_offsets = torch.stack([
            target_x - pred_x,
            target_y - pred_y
        ], dim=2)  # [batch, num_keypoints, 2]

        
        
        # huber loss on offsets
        target_flat_orig = target_heatmaps.view(batch_size, num_keypoints, -1)
        valid_mask = (target_flat_orig.sum(dim=2) > 0).unsqueeze(-1)  # [batch, num_keypoints, 1]
        
        if valid_mask.sum() > 0:  # if valid kps (many frames have like not many keypoints)
            valid_pred_offsets = pred_offsets_at_points[valid_mask.expand_as(pred_offsets_at_points)]
            valid_target_offsets = target_offsets[valid_mask.expand_as(target_offsets)]
            #print(valid_target_offsets[0])
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
    
    # predicted offsets at predicted locations
    pred_y_idx = (pred_y / scale_factor).long().clamp(0, h-1)
    pred_x_idx = (pred_x / scale_factor).long().clamp(0, w-1)
    
    # batch indices for gathering offsets
    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_keypoints)
    keypoint_indices = torch.arange(num_keypoints).view(1, -1).expand(batch_size, -1)
    
    # offsets at predicted locations
    pred_offsets_at_points = offsets[batch_indices, keypoint_indices, :, pred_y_idx, pred_x_idx]

    #print(pred_offsets_at_points[0])
    
    # apply offsets to get refined coordinates
    refined_x = pred_x + pred_offsets_at_points[:, :, 0]
    refined_y = pred_y + pred_offsets_at_points[:, :, 1]
    
    # new coordinates 
    coordinates = torch.stack([refined_x, refined_y], dim=2)
    
    return coordinates

def extract_coordinates_from_target_heatmaps(target_heatmaps):
    """Extract ground truth coordinates from target heatmaps"""
    batch_size, num_keypoints, h, w = target_heatmaps.shape
    
    target_flat = target_heatmaps.view(batch_size, num_keypoints, -1)
    _, target_idx = torch.max(target_flat, dim=2)  # [batch, num_keypoints]
    
    scale_factor = 256.0 / 32.0  
    target_y = (target_idx // w).float() * scale_factor
    target_x = (target_idx % w).float() * scale_factor
    
    # Stack coordinates (B, num_keypoints, 2)
    coordinates = torch.stack([target_x, target_y], dim=2)
    
    return coordinates

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
    valid_keypoints = ~((gt_coords[:, :, 0] == 0) & (gt_coords[:, :, 1] == 0)) 
    
    # threshold 
    threshold_distances = nose_eye_dist_modified.unsqueeze(1) 
    correct = (distances < 0.3 * threshold_distances) & valid_keypoints  # (B, 22)
    total_valid = valid_keypoints.sum().item()
    total_correct = correct.sum().item()
    
    if total_valid == 0:
        return 0.0, 0
    
    pck = total_correct / total_valid
    return pck, total_valid

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on data with both loss and PCK metrics"""
    model.eval()
    total_loss = 0
    total_scoremap_loss = 0
    total_loc_loss = 0
    num_batches = len(data_loader)
    
    # pck values
    all_pck_scores = []
    total_valid_keypoints = 0
    thresholds = [0.3]
    pck_results = {th: [] for th in thresholds}
    
    print("Evaluating model on training data...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            
            pred_scoremaps, pred_offsets = model(images)
            loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
            
            total_loss += loss.item()
            total_scoremap_loss += scoremap_loss.item()
            total_loc_loss += loc_loss.item()
            
            # coordinates for pck
            pred_coords = extract_coordinates_from_heatmaps(pred_scoremaps, pred_offsets)
            gt_coords = extract_coordinates_from_target_heatmaps(target_heatmaps)  
            #print(gt_coords[0])
            
            # pck for different thresholds (just using 0.3 for now since that's what the paper uses)
            for threshold in thresholds:
                pck_score, valid_count = compute_pck(pred_coords, gt_coords, threshold)
                pck_results[threshold].append((pck_score, valid_count))
    
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
    parser.add_argument('--checkpoint', default='final_checkpoint.pth', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    split_file = os.path.join(root_dir, f'TrainTestInfo_shuffle{args.shuffle}.csv')
    
    # train dataloader (also can check perfomrance on train set)
    train_loader, _ = get_horse_dataloaders(
        root_dir=root_dir,
        split_file=split_file,
        batch_size=args.batch_size,
        shuffle=False
    )

    # within + across domain (i.e. inside or outside train distribution)
    within_domain_loader, across_domain_loader = get_horse_domain_dataloaders(
        root_dir=root_dir,
        split_file=split_file,
        batch_size=args.batch_size
    )
        
    print(f"Train dataset size: {len(train_loader.dataset)}")
    
    # model
    model = KeypointResNet50(num_keypoints=22).to(device)
    
    checkpoint_path = os.path.join(root_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # criterion (same as training)
    criterion = CombinedKeypointLoss(loc_weight=0.05)
    
    # evaluate
    avg_loss, avg_scoremap_loss, avg_loc_loss, pck_scores, total_valid = evaluate_model(model, within_domain_loader, criterion, device)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (TRAINING SET)")
    print("="*50)
    print(f'Train Loss: {avg_loss:.4f}')
    print(f'  Scoremap Loss: {avg_scoremap_loss:.4f}')
    print(f'  Location Loss: {avg_loc_loss:.4f}')
    print()
    print("PCK Scores:")
    for threshold, pck in pck_scores.items():
        print(f'  PCK@{threshold:.2f}: {pck:.3f}')
    print("="*50)

if __name__ == '__main__':
    main() 