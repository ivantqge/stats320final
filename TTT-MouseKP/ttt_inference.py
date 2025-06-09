import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from models.ResNet import ResNetMouse
from models.SSHead import extractor_from_layer2, head_on_layer2, ExtractorHead

def load_ttt_model(model_path, depth=26, width=1, num_keypoints=5, kernel_size=3):
    """Load model with SSH components for TTT"""

    # loading model
    net = ResNetMouse(depth=depth, width=width, num_keypoints=num_keypoints, kernel_size=kernel_size).cuda()
    
    # ssh components 
    ext = extractor_from_layer2(net)
    head = head_on_layer2(net, width, 4)
    ssh = ExtractorHead(ext, head).cuda()
    
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    head.load_state_dict(checkpoint['head'])
    
    return net, ssh, head

def load_regular_model(model_path, depth=26, width=1, num_keypoints=5, kernel_size=3):
    """Load regular model without SSH components"""
    
    # model trained w/o ssh 

    net = ResNetMouse(depth=depth, width=width, num_keypoints=num_keypoints, kernel_size=kernel_size).cuda()
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    return net

def process_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(image).unsqueeze(0).cuda()
    return img_tensor, image, original_size

def ttt_adapt(net, ssh, head, img_tensor, num_steps=0, lr=0.001):
    """Test-Time Training: adapt model using self-supervised task - original parameters"""

    # optimize ssh.parameters() (extractor + head)
    optimizer_ssh = optim.SGD(ssh.parameters(), lr=lr)  
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    
    ssh.train()
    
    for step in range(num_steps):
        optimizer_ssh.zero_grad()
        
        # quadrant prediction task
        B, C, H, W = img_tensor.shape
        
        patches = []
        patch_labels = []
        
        for b in range(B):
            # 4 quadrants
            q1 = img_tensor[b, :, :H//2, :W//2]
            q2 = img_tensor[b, :, :H//2, W//2:]
            q3 = img_tensor[b, :, H//2:, :W//2]
            q4 = img_tensor[b, :, H//2:, W//2:]

            # ensure all quadrants are the same size
            min_H = min(q1.size(1), q2.size(1), q3.size(1), q4.size(1))
            min_W = min(q1.size(2), q2.size(2), q3.size(2), q4.size(2))
            
            q1 = nn.functional.adaptive_avg_pool2d(q1.unsqueeze(0), (min_H, min_W)).squeeze(0)
            q2 = nn.functional.adaptive_avg_pool2d(q2.unsqueeze(0), (min_H, min_W)).squeeze(0)
            q3 = nn.functional.adaptive_avg_pool2d(q3.unsqueeze(0), (min_H, min_W)).squeeze(0)
            q4 = nn.functional.adaptive_avg_pool2d(q4.unsqueeze(0), (min_H, min_W)).squeeze(0)
            
            # create patches with random ordering
            quadrants = [q1, q2, q3, q4]
            order = torch.randperm(4)
            
            for idx in order:
                patches.append(quadrants[idx])
                patch_labels.append(idx)
        
        patches = torch.stack(patches).cuda()
        patch_labels = torch.tensor(patch_labels).cuda()
        
        # forward + backward pass on ssh task 

        outputs_ssh = ssh(patches)
        loss_ssh = criterion_ssh(outputs_ssh, patch_labels)
        loss_ssh.backward()
        optimizer_ssh.step()

        print(f"  Step {step+1}: SSH loss = {loss_ssh.item():.4f}")
    
    ssh.eval()

def load_ground_truth_keypoints(image_path):
    """Load ground truth keypoints from mask files"""

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_dir = image_path.replace('val_blur', 'val/masks').replace(base_name + '.png', '')
    
    gt_coords = []
    
    for kp_idx in range(5):  # 5 keypoints
        mask_path = os.path.join(mask_dir, f'{base_name}_kp{kp_idx}.png')
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            
            # using pixel w/ highest probability as keypoint coordinates
            y, x = np.unravel_index(mask_np.argmax(), mask_np.shape)

            gt_coords.extend([x, y]) 
            
        else:
            gt_coords.extend([None, None])
    
    return gt_coords

def load_ground_truth_heatmaps(image_path):
    """Load ground truth heatmaps from mask files"""

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_dir = image_path.replace('val_corrupted', 'val/masks').replace(base_name + '.png', '')
    
    gt_heatmaps = []
    
    for kp_idx in range(5):  # 5 keypoints
        mask_path = os.path.join(mask_dir, f'{base_name}_kp{kp_idx}.png')
        
        if os.path.exists(mask_path):

            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask).astype(np.float32)
            if mask_np.max() > 0:
                mask_np = mask_np / mask_np.max()
            gt_heatmaps.append(mask_np)

        else:
            gt_heatmaps.append(np.zeros((384, 416), dtype=np.float32))  # random default size, they all have heatmaps
    
    return torch.tensor(np.stack(gt_heatmaps), dtype=torch.float32)


def calculate_pck(pred_coords, gt_coords, image_size, threshold=0.05):

    width, height = image_size
    diagonal = np.sqrt(width**2 + height**2)
    threshold_pixels = threshold * diagonal
    
    correct_keypoints = 0
    valid_keypoints = 0
    
    for i in range(0, len(gt_coords), 2):
        x_gt, y_gt = gt_coords[i], gt_coords[i+1]
        x_pred, y_pred = pred_coords[i], pred_coords[i+1]
        
        if x_gt != 0 or y_gt != 0:
            distance = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
            if distance < threshold_pixels:
                correct_keypoints += 1
            valid_keypoints += 1
    
    pck = correct_keypoints / valid_keypoints if valid_keypoints > 0 else 0
    return pck, correct_keypoints, valid_keypoints

def run_inference_comparison(regular_model_path, ttt_model_path, test_image_path, output_dir):
    
    '''inference comparison between TTT and regular model without TTT'''
    
    # load models
    regular_net = load_regular_model(regular_model_path)
    ttt_net, ssh_net, ssh_head = load_ttt_model(ttt_model_path)
    
    # load ground truth data (both coordinates and heatmaps)
    print("Loading ground truth keypoints from masks...")
    print(test_image_path)
    gt_coords_list = load_ground_truth_keypoints(test_image_path)
    gt_heatmaps = load_ground_truth_heatmaps(test_image_path)
    
    # use corrupted image (can also use regular images in val folder)
    corrupt_tensor, corrupt_image, corrupt_size = process_image(test_image_path)
    
    # no ttt coords
    with torch.no_grad():
        corrupt_heatmaps = regular_net(corrupt_tensor)
        corrupt_coords = regular_net.get_keypoint_coordinates(corrupt_heatmaps)
    
    corrupt_coords_list = corrupt_coords[0].cpu().numpy()

    # ttt coords

    ttt_adapt(ttt_net, ssh_net, ssh_head, corrupt_tensor, num_steps=20, lr=0.01)
    
    with torch.no_grad():
        ttt_heatmaps = ttt_net(corrupt_tensor)
        ttt_coords = ttt_net.get_keypoint_coordinates(ttt_heatmaps)
    
    ttt_coords_list = ttt_coords[0].cpu().numpy()
        
    # convert to pixel coordinates
    def coords_to_pixels(coords_list, width, height):
        coords_pixels = []
        for i in range(0, len(coords_list), 2):
            x_norm, y_norm = coords_list[i], coords_list[i+1]
            pixel_x = x_norm * width
            pixel_y = y_norm * height
            coords_pixels.append((pixel_x, pixel_y))
        return coords_pixels
    
    def gt_coords_to_tuples(gt_coords_list):
        gt_pixels = []
        for i in range(0, len(gt_coords_list), 2):
            x, y = gt_coords_list[i], gt_coords_list[i+1]
            gt_pixels.append((x, y))
        return gt_pixels
    
    width, height = corrupt_size
    gt_coords_pixels = gt_coords_to_tuples(gt_coords_list)
    corrupt_coords_pixels = coords_to_pixels(corrupt_coords_list, width, height)
    ttt_coords_pixels = coords_to_pixels(ttt_coords_list, width, height)
    
    print("\n=== COORDINATE-BASED ERROR ANALYSIS ===")
    print("Pixel distance error (Ground Truth vs No TTT):")
    total_corrupt_error = 0
    valid_keypoints = 0
    
    for i, ((x_gt, y_gt), (x_corrupt, y_corrupt)) in enumerate(zip(gt_coords_pixels, corrupt_coords_pixels)):
        if x_gt != 0 or y_gt != 0:
            diff_x = abs(x_corrupt - x_gt)
            diff_y = abs(y_corrupt - y_gt)
            diff_total = (diff_x**2 + diff_y**2)**0.5
            total_corrupt_error += diff_total
            valid_keypoints += 1
            print(f"  kp{i}: Delta x={diff_x:.1f}, Delta y={diff_y:.1f}, total={diff_total:.1f}px")
        else:
            print(f"  kp{i}: (no ground truth)")
    
    avg_corrupt_error = total_corrupt_error / valid_keypoints if valid_keypoints > 0 else 0
    print(f"Average pixel error (no TTT): {avg_corrupt_error:.1f}px")
    
    print("\nPixel distance error (Ground Truth vs TTT):")
    total_ttt_error = 0
    
    for i, ((x_gt, y_gt), (x_ttt, y_ttt)) in enumerate(zip(gt_coords_pixels, ttt_coords_pixels)):

        if x_gt != 0 or y_gt != 0:
            diff_x = abs(x_ttt - x_gt)
            diff_y = abs(y_ttt - y_gt)
            diff_total = (diff_x**2 + diff_y**2)**0.5
            total_ttt_error += diff_total
            print(f"  kp{i}: Delta x={diff_x:.1f}, Delta y={diff_y:.1f}, total={diff_total:.1f}px")

        else:

            print(f"  kp{i}: (no ground truth)")
    
    avg_ttt_error = total_ttt_error / valid_keypoints if valid_keypoints > 0 else 0
    print(f"Average pixel error (TTT): {avg_ttt_error:.1f}px")
    
    # calculate improvement/degradation
    pixel_improvement = total_corrupt_error - total_ttt_error
    print(f"Pixel error {'IMPROVEMENT' if pixel_improvement > 0 else 'DEGRADATION'}: {pixel_improvement:.1f}px")
    
    # pck
    corrupt_pck, corrupt_correct, _ = calculate_pck(
        [coord for pair in corrupt_coords_pixels for coord in pair], 
        gt_coords_list, corrupt_size
    )
    ttt_pck, ttt_correct, _ = calculate_pck(
        [coord for pair in ttt_coords_pixels for coord in pair], 
        gt_coords_list, corrupt_size
    )
    
    print(f"\nPCK:")
    print(f"  Corrupted (Regular Model): {corrupt_pck:.3f} ({corrupt_correct}/{valid_keypoints})")
    print(f"  TTT: {ttt_pck:.3f} ({ttt_correct}/{valid_keypoints})")
    pck_improvement = ttt_pck - corrupt_pck
    print(f"  PCK {'IMPROVEMENT' if pck_improvement > 0 else 'DEGRADATION'}: {pck_improvement:.3f}")
    
    # visualization nice plot

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # convert image to grayscale for display
    corrupt_image_gray = corrupt_image.convert('L') if hasattr(corrupt_image, 'convert') else corrupt_image
    
    # gt kps
    axes[0].imshow(corrupt_image_gray, cmap='gray')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (x, y) in enumerate(gt_coords_pixels):
        # Only plot if not at origin (invalid keypoint)
        if x != 0 or y != 0:
            axes[0].plot(x, y, 'o', color=colors[i], markersize=12, markeredgecolor='white', markeredgewidth=3)
            axes[0].text(x+5, y-5, f'kp{i}', color=colors[i], fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    axes[0].set_title('Ground Truth Keypoints', fontsize=14)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    
    # corrupted w/ both regular and ttt predictions
    axes[1].imshow(corrupt_image_gray, cmap='gray')
    for i, ((x_gt, y_gt), (x_corrupt, y_corrupt)) in enumerate(zip(gt_coords_pixels, corrupt_coords_pixels)):

        if x_gt != 0 or y_gt != 0:
            axes[1].plot(x_corrupt, y_corrupt, 's', color=colors[i], markersize=8, alpha=0.7, markeredgecolor='white', markeredgewidth=0)
    
    for i, ((x_gt, y_gt), (x_ttt, y_ttt)) in enumerate(zip(gt_coords_pixels, ttt_coords_pixels)):

        if x_gt != 0 or y_gt != 0:
            axes[1].plot(x_ttt, y_ttt, 'o', color=colors[i], markersize=10, markeredgecolor='white', markeredgewidth=0)
    
    axes[1].set_title('Corrupted Image: Regular vs TTT', fontsize=14)
    axes[1].set_xlim(0, width)
    axes[1].set_ylim(height, 0)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(test_image_path))[0]
    output_path = os.path.join(output_dir, f'ttt_comparison_{image_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved to: {output_path}")
    
    return {
        'ground_truth': gt_coords_pixels,
        'corrupted': corrupt_coords_pixels,
        'ttt': ttt_coords_pixels,
        'pixel_error_corrupt': avg_corrupt_error,
        'pixel_error_ttt': avg_ttt_error,
        'pixel_improvement': pixel_improvement,
        'pck_corrupt': corrupt_pck,
        'pck_ttt': ttt_pck,
        'pck_improvement': pck_improvement
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regular_model', default='./results/mouse_tracking_no_ssh/model_final.pth')
    parser.add_argument('--ttt_model', default='./results/mouse_tracking/model_final.pth')
    parser.add_argument('--test_image', required=True, help='Path to corrupted test image')
    parser.add_argument('--output_dir', default='./results/ttt_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_inference_comparison(args.regular_model, args.ttt_model, args.test_image, args.output_dir)

if __name__ == '__main__':
    main() 