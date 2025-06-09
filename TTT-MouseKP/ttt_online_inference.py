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
import glob
import torch.nn.functional as F
from models.ResNet import ResNetMouse
from models.SSHead import extractor_from_layer2, head_on_layer2, ExtractorHead

def load_ttt_model(model_path, depth=26, width=1, num_keypoints=5, kernel_size=3):
    """Load model with SSH components for TTT"""
    
    net = ResNetMouse(depth=depth, width=width, num_keypoints=num_keypoints, kernel_size=kernel_size).cuda()
    
    ext = extractor_from_layer2(net)
    head = head_on_layer2(net, width, 4)
    ssh = ExtractorHead(ext, head).cuda()
    
    # weights
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    head.load_state_dict(checkpoint['head'])
    
    return net, ssh, head

def load_regular_model(model_path, depth=26, width=1, num_keypoints=5, kernel_size=3):
    """Load regular model without SSH components"""

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

def ttt_adapt_online(net, ssh, optimizer_ssh, img_tensor, num_steps=5, criterion_ssh=None):
    """Online TTT adaptation: adapt model using self-supervised task"""
    if criterion_ssh is None:
        criterion_ssh = nn.CrossEntropyLoss().cuda()
    
    # ssh to train mode
    ssh.train()
    
    total_loss = 0.0
    
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
        
        total_loss += loss_ssh.item()
    
    # back to eval mode
    ssh.eval()
    
    return total_loss / num_steps

def load_ground_truth_keypoints(image_path):
    """Load ground truth keypoints from mask files"""

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    mask_dir = 'data/mouse_dataset/val/masks/'
    gt_coords = []
    
    for kp_idx in range(5):  # 5 keypoints
        mask_path = os.path.join(mask_dir, f'{base_name}_kp{kp_idx}.png')
        
        if os.path.exists(mask_path):
            # load mask and find keypoint location
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            
            # brightest pixel/highest probability pixel (keypoint location)
            y, x = np.unravel_index(mask_np.argmax(), mask_np.shape)
            gt_coords.extend([x, y])  # x, y coordinates
        else:
            # if mask doesn't exist, append None coordinates
            gt_coords.extend([None, None])
    
    return gt_coords

def calculate_pck(pred_coords, gt_coords, image_size, threshold=0.05):
    """Calculate Percentage of Correct Keypoints (PCK)"""
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

def visualize_predictions(img_pil, img_size, gt_coords, 
                         regular_heatmaps, regular_coords, 
                         ttt_heatmaps_before, ttt_coords_before,
                         ttt_heatmaps_after, ttt_coords_after,
                         image_name, output_dir, ssh_loss):
    """simple side-by-side visualization comparing ground truth and predictions"""
    
    # convert PIL image to numpy for plotting
    img_np = np.array(img_pil) / 255.0
    width, height = img_size
    
    # figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # left plot: ground truth
    axes[0].imshow(img_np)
    axes[0].set_title(f"Ground Truth\n{image_name}")
    axes[0].axis('off')
    
    for kp_idx in range(5):  # 5 keypoints
        x_gt = gt_coords[kp_idx * 2]
        y_gt = gt_coords[kp_idx * 2 + 1]
        if x_gt != 0 or y_gt != 0:  # Valid keypoint
            axes[0].plot(x_gt, y_gt, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            axes[0].text(x_gt + 5, y_gt - 5, f'{kp_idx}', color='red', fontweight='bold', fontsize=12)
    
    # right plot is regular vs. ttt
    axes[1].imshow(img_np)
    axes[1].set_title(f"Regular Model vs TTT Model\nSSH Loss: {ssh_loss:.3f}")
    axes[1].axis('off')
    
    for kp_idx in range(5):
        x_regular = regular_coords[0][kp_idx * 2].item() * width  # Convert from normalized
        y_regular = regular_coords[0][kp_idx * 2 + 1].item() * height
        axes[1].plot(x_regular, y_regular, 'bs', markersize=8, markeredgecolor='white', markeredgewidth=1, label='Regular' if kp_idx == 0 else "")
        axes[1].text(x_regular + 5, y_regular - 5, f'{kp_idx}', color='blue', fontweight='bold', fontsize=10)
    
    for kp_idx in range(5):
        x_ttt = ttt_coords_after[0][kp_idx * 2].item() * width  # Convert from normalized
        y_ttt = ttt_coords_after[0][kp_idx * 2 + 1].item() * height
        axes[1].plot(x_ttt, y_ttt, 'g^', markersize=8, markeredgecolor='white', markeredgewidth=1, label='TTT' if kp_idx == 0 else "")
        axes[1].text(x_ttt + 5, y_ttt - 5, f'{kp_idx}', color='green', fontweight='bold', fontsize=10)
    
    axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    image_name_no_ext = os.path.splitext(image_name)[0]  # Remove extension
    vis_path = os.path.join(output_dir, f'visualization_{image_name_no_ext}.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return vis_path

def run_online_ttt(regular_model_path, ttt_model_path, test_images_dir, output_dir, num_steps=5, lr=0.001, visualize=True):
    """Run online TTT across multiple images without resetting"""
    
    print(f"Running online TTT on images in: {test_images_dir}")
    print(f"Adaptation steps per image: {num_steps}, LR: {lr}")
    
    regular_net = load_regular_model(regular_model_path)
    ttt_net, ssh_net, ssh_head = load_ttt_model(ttt_model_path)
    
    # ssh optimizer to train
    optimizer_ssh = optim.SGD(ssh_net.parameters(), lr=lr)
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    test_images.sort() # in order
    
    if len(test_images) == 0:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    results = {
        'image_names': [],
        'regular_pck': [],
        'ttt_pck': [],
        'regular_pixel_error': [],
        'ttt_pixel_error': [],
        'ssh_loss': []
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING ONLINE TTT ADAPTATION")
    print("="*80)
    
    for i, image_path in enumerate(test_images):
        image_name = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(test_images)}] Processing: {image_name}")
        
        gt_coords_list = load_ground_truth_keypoints(image_path)
        if None in gt_coords_list:
            print(f"  Skipping {image_name} - missing ground truth")
            continue
        
        img_tensor, img_pil, img_size = process_image(image_path)
        
        # regular model
        with torch.no_grad():
            regular_heatmaps = regular_net(img_tensor)
            regular_coords = regular_net.get_keypoint_coordinates(regular_heatmaps)
        
        regular_coords_list = regular_coords[0].cpu().numpy()
        
        with torch.no_grad():
            ttt_heatmaps_before = ttt_net(img_tensor)
            ttt_coords_before = ttt_net.get_keypoint_coordinates(ttt_heatmaps_before)
        
        # TTT online
        avg_ssh_loss = ttt_adapt_online(ttt_net, ssh_net, optimizer_ssh, img_tensor, 
                                       num_steps=num_steps, criterion_ssh=criterion_ssh)
        
        with torch.no_grad():
            ttt_heatmaps_after = ttt_net(img_tensor)
            ttt_coords_after = ttt_net.get_keypoint_coordinates(ttt_heatmaps_after)
        
        ttt_coords_list = ttt_coords_after[0].cpu().numpy()
        
        vis_path = None
        if visualize:
            vis_path = visualize_predictions(
                img_pil, img_size, gt_coords_list,
                regular_heatmaps, regular_coords,
                ttt_heatmaps_before, ttt_coords_before,
                ttt_heatmaps_after, ttt_coords_after,
                image_name, output_dir, avg_ssh_loss
            )
        
        # convert to pixel coordinates
        width, height = img_size
        
        def coords_to_pixels(coords_list, w, h):
            pixels = []
            for i in range(0, len(coords_list), 2):
                x_norm, y_norm = coords_list[i], coords_list[i+1]
                pixels.extend([x_norm * w, y_norm * h])
            return pixels
        
        regular_pixels = coords_to_pixels(regular_coords_list, width, height)
        ttt_pixels = coords_to_pixels(ttt_coords_list, width, height)
        
        # pck
        regular_pck, _, _ = calculate_pck(regular_pixels, gt_coords_list, img_size)
        ttt_pck, _, _ = calculate_pck(ttt_pixels, gt_coords_list, img_size)
        
        # calculate pixel errors
        def calc_avg_pixel_error(pred_pixels, gt_coords):
            total_error = 0
            valid_kp = 0
            for j in range(0, len(gt_coords), 2):
                x_gt, y_gt = gt_coords[j], gt_coords[j+1]
                if x_gt != 0 or y_gt != 0:
                    x_pred, y_pred = pred_pixels[j], pred_pixels[j+1]
                    error = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
                    total_error += error
                    valid_kp += 1
            return total_error / valid_kp if valid_kp > 0 else 0
        
        regular_error = calc_avg_pixel_error(regular_pixels, gt_coords_list)
        ttt_error = calc_avg_pixel_error(ttt_pixels, gt_coords_list)
        
        # store results
        results['image_names'].append(image_name)
        results['regular_pck'].append(regular_pck)
        results['ttt_pck'].append(ttt_pck)
        results['regular_pixel_error'].append(regular_error)
        results['ttt_pixel_error'].append(ttt_error)
        results['ssh_loss'].append(avg_ssh_loss)
        
        # progress for each image
        print(f"  SSH Loss: {avg_ssh_loss:.4f}")
        print(f"  PCK - Regular: {regular_pck:.3f}, TTT: {ttt_pck:.3f} (Δ: {ttt_pck-regular_pck:+.3f})")
        print(f"  Pixel Error - Regular: {regular_error:.1f}px, TTT: {ttt_error:.1f}px (Δ: {ttt_error-regular_error:+.1f}px)")
        if vis_path:
            print(f"  Visualization saved to: {vis_path}")
    
    # final summary
    print("\n" + "="*80)
    print("ONLINE TTT RESULTS SUMMARY")
    print("="*80)
    
    if len(results['regular_pck']) > 0:
        avg_regular_pck = np.mean(results['regular_pck'])
        avg_ttt_pck = np.mean(results['ttt_pck'])
        avg_regular_error = np.mean(results['regular_pixel_error'])
        avg_ttt_error = np.mean(results['ttt_pixel_error'])
        
        print(f"Average PCK:")
        print(f"  Regular Model: {avg_regular_pck:.3f}")
        print(f"  Online TTT:    {avg_ttt_pck:.3f}")
        print(f"  Correctness Change:   {avg_ttt_pck - avg_regular_pck:+.3f}")
        
        print(f"\nAverage Pixel Error:")
        print(f"  Regular Model: {avg_regular_error:.1f}px")
        print(f"  Online TTT:    {avg_ttt_error:.1f}px")
        print(f"  Error Change:   {avg_ttt_error - avg_regular_error:+.1f}px")
        
        
        # Create performance over time plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        x_indices = range(1, len(results['image_names']) + 1)
        
        # PCK over time
        axes[0,0].plot(x_indices, results['regular_pck'], 'o-', label='Regular Model', color='red', alpha=0.7)
        axes[0,0].plot(x_indices, results['ttt_pck'], 'o-', label='Online TTT', color='blue', alpha=0.7)
        axes[0,0].set_xlabel('Image Index')
        axes[0,0].set_ylabel('PCK@5%')
        axes[0,0].set_title('PCK Performance Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Pixel error over time
        axes[0,1].plot(x_indices, results['regular_pixel_error'], 'o-', label='Regular Model', color='red', alpha=0.7)
        axes[0,1].plot(x_indices, results['ttt_pixel_error'], 'o-', label='Online TTT', color='blue', alpha=0.7)
        axes[0,1].set_xlabel('Image Index')
        axes[0,1].set_ylabel('Pixel Error')
        axes[0,1].set_title('Pixel Error Over Time')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # SSH loss over time
        axes[1,0].plot(x_indices, results['ssh_loss'], 'o-', color='green', alpha=0.7)
        axes[1,0].set_xlabel('Image Index')
        axes[1,0].set_ylabel('SSH Loss')
        axes[1,0].set_title('Self-Supervised Loss Over Time')
        axes[1,0].grid(True, alpha=0.3)
        
        # cumulative improvement over all test images
        pck_improvements = [ttt - reg for ttt, reg in zip(results['ttt_pck'], results['regular_pck'])]
        cumulative_pck_improvement = np.cumsum(pck_improvements) / np.arange(1, len(pck_improvements) + 1)
        
        axes[1,1].plot(x_indices, cumulative_pck_improvement, 'o-', color='purple', alpha=0.7)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Image Index')
        axes[1,1].set_ylabel('Cumulative PCK Improvement')
        axes[1,1].set_title('Cumulative TTT Improvement')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'online_ttt_performance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        results_path = os.path.join(output_dir, 'online_ttt_results.txt')
        with open(results_path, 'w') as f:
            f.write("Online TTT Results\n")
            f.write("==================\n\n")
            f.write(f"Number of images: {len(results['image_names'])}\n")
            f.write(f"Adaptation steps per image: {num_steps}\n")
            f.write(f"Learning rate: {lr}\n\n")
            f.write(f"Average PCK - Regular: {avg_regular_pck:.3f}\n")
            f.write(f"Average PCK - Online TTT: {avg_ttt_pck:.3f}\n")
            f.write(f"PCK Improvement: {avg_ttt_pck - avg_regular_pck:+.3f}\n\n")
            f.write(f"Average Pixel Error - Regular: {avg_regular_error:.1f}px\n")
            f.write(f"Average Pixel Error - Online TTT: {avg_ttt_error:.1f}px\n")
            f.write(f"Pixel Error Improvement: {avg_ttt_error - avg_regular_error:+.1f}px\n\n")
            f.write("Per-Image Results:\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(results['image_names']):
                f.write(f"{name}: PCK {results['regular_pck'][i]:.3f}->{results['ttt_pck'][i]:.3f}, "
                       f"Error {results['regular_pixel_error'][i]:.1f}->{results['ttt_pixel_error'][i]:.1f}px\n")
        
        print(f"\nResults saved to:")
        print(f"  Plot: {plot_path}")
        print(f"  Details: {results_path}")
    
    else:
        print("No valid images processed!")

def main():
    parser = argparse.ArgumentParser(description='Online TTT Inference - Continuous adaptation across multiple images')
    parser.add_argument('--regular_model', default='./results/mouse_tracking_no_ssh/model_final.pth',
                       help='Path to regular model without SSH')
    parser.add_argument('--ttt_model', default='./results/mouse_tracking/model_final.pth',
                       help='Path to TTT model with SSH')
    parser.add_argument('--test_images_dir', required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', default='./results/online_ttt_results',
                       help='Output directory for results')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of TTT adaptation steps per image')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for TTT adaptation')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate heatmap comparison visualizations for each image')
    
    args = parser.parse_args()
    
    run_online_ttt(args.regular_model, args.ttt_model, args.test_images_dir, 
                   args.output_dir, args.num_steps, args.lr, args.visualize)

if __name__ == '__main__':
    main() 