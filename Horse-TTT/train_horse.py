import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from horse_dataloader import get_horse_dataloaders
from horse_model import KeypointResNet50

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
        
        # huber loss w/ offsets
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

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_scoremap_loss = 0
    total_loc_loss = 0
    num_batches = len(train_loader)
    
    for batch in tqdm(train_loader, desc='Training'):
        images = batch['image'].to(device)
        target_heatmaps = batch['heatmaps'].to(device)
        
        optimizer.zero_grad()
        pred_scoremaps, pred_offsets = model(images)
        
        loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_scoremap_loss += scoremap_loss.item()
        total_loc_loss += loc_loss.item()
    
    avg_loss = total_loss / num_batches
    avg_scoremap_loss = total_scoremap_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    
    return avg_loss, avg_scoremap_loss, avg_loc_loss

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_scoremap_loss = 0
    total_loc_loss = 0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            
            pred_scoremaps, pred_offsets = model(images)
            loss, scoremap_loss, loc_loss = criterion(pred_scoremaps, pred_offsets, target_heatmaps)
            
            total_loss += loss.item()
            total_scoremap_loss += scoremap_loss.item()
            total_loc_loss += loc_loss.item()
    
    avg_loss = total_loss / num_batches
    avg_scoremap_loss = total_scoremap_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    
    return avg_loss, avg_scoremap_loss, avg_loc_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--test_time_training', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    split_file = os.path.join(root_dir, f'TrainTestInfo_shuffle{args.shuffle}.csv')
    
    train_loader, test_loader = get_horse_dataloaders(
        root_dir=root_dir,
        split_file=split_file,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # model w/ all 22 kps
    model = KeypointResNet50(num_keypoints=22).to(device)
    criterion = CombinedKeypointLoss(loc_weight=0.05)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_losses = train(model, train_loader, criterion, optimizer, device)
        # if epoch % 10 == 0:
        #     test_losses = test(model, test_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_losses[0]:.4f} (Scoremap: {train_losses[1]:.4f}, Loc: {train_losses[2]:.4f})')
        # if epoch % 10 == 0:
        #     print(f'Test Loss: {test_losses[0]:.4f} (Scoremap: {test_losses[1]:.4f}, Loc: {test_losses[2]:.4f})')
        
        # if test_losses[0] < best_loss:
        #     best_loss = test_losses[0]
        #     torch.save({
        #         'state_dict': model.state_dict(),
        #         'get_keypoint_coordinates': model.get_keypoint_coordinates
        #     }, 'best_model.pth')
        #     print('Saved best model')
        
        print('-' * 50)
    
    # saving final checkpt
    checkpoint_path = os.path.join(root_dir, 'final_checkpoint.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[0],
        'get_keypoint_coordinates': model.get_keypoint_coordinates
    }, checkpoint_path)
    print(f'Saved final checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    main() 