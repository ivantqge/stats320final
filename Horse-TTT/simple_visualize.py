import torch
import matplotlib.pyplot as plt
import numpy as np
from horse_dataloader import get_horse_dataloaders, get_horse_domain_dataloaders
import torch.nn.functional as F
from horse_model import KeypointResNet50
from keypoint_jigsaw_model import KeypointJigsawModel

def visualize_sample():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load regular horse model
    regular_model = KeypointResNet50(num_keypoints=22)
    regular_checkpoint = torch.load('final_checkpoint.pth', map_location=device)
    regular_model.load_state_dict(regular_checkpoint['model_state_dict'])
    regular_model.to(device)
    regular_model.eval()
    
    # load jigsaw model 
    jigsaw_model = KeypointJigsawModel.load_from_checkpoint('final_keypoint_jigsaw_checkpoint_sigma0.5.pth', device)
    jigsaw_model.to(device)
    jigsaw_model.eval()
    
    # get data
    train_loader, test_loader = get_horse_dataloaders(
        root_dir='/home/ubuntu/ttt_cifar_release/horse10',
        split_file='/home/ubuntu/ttt_cifar_release/horse10/TrainTestInfo_shuffle1.csv',
        batch_size=16,
        shuffle=True
    )

    within_domain_loader, across_domain_loader = get_horse_domain_dataloaders(
        root_dir='/home/ubuntu/ttt_cifar_release/horse10',
        split_file='/home/ubuntu/ttt_cifar_release/horse10/TrainTestInfo_shuffle1.csv',
        batch_size=32
    )

    dataloader = across_domain_loader
    i = 0

    # random sample
    for batch in dataloader:
        if i < 2:
            i += 1
            continue
        image = batch['image'][10]  # Remove batch dimension
        keypoints = batch['keypoints'][10]  # Remove batch dimension
        heatmaps = batch['heatmaps'][10]  # Remove batch dimension
        filename = batch['path'][10] if 'path' in batch else "unknown"
        
        print(f"Image shape: {image.shape}")
        
        print(f"Keypoints shape: {keypoints.shape}")
        print(keypoints)
        print(f"Heatmaps shape: {heatmaps.shape}")
        print(f"Filename: {filename}")
        
        # denorm from resnet params
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        
        image_input = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # regular model prediction
            regular_scoremaps, regular_offsets = regular_model(image_input)
            regular_heatmaps = regular_scoremaps[0].cpu()  # Remove batch dimension
            
            # jigsaw model prediction  
            jigsaw_scoremaps, jigsaw_offsets = jigsaw_model.forward_keypoints(image_input)
            jigsaw_heatmaps = jigsaw_scoremaps[0].cpu()  # Remove batch dimension
            
            regular_heatmaps = torch.sigmoid(regular_heatmaps)
            jigsaw_heatmaps = torch.sigmoid(jigsaw_heatmaps)
            
        
        fig, axes = plt.subplots(3, 7, figsize=(28, 12))
        
        # original image in first column of each row
        for row in range(3):
            axes[row, 0].imshow(image_np)
            axes[row, 0].axis('off')
        axes[0, 0].set_title("Original Image")
        axes[1, 0].set_title("Original Image")  
        axes[2, 0].set_title("Original Image")
        
        # first 6 heatmaps for each model
        all_heatmaps = [
            (heatmaps, "Ground Truth"),
            (regular_heatmaps, "Regular Model"), 
            (jigsaw_heatmaps, "TTT Model")
        ]
        
        for row, (model_heatmaps, model_name) in enumerate(all_heatmaps):
            for i in range(6):
                if i < model_heatmaps.shape[0]:
                    col = i + 1
                    
                    # resize heatmap to image size for overlay
                    heatmap_resized = F.interpolate(
                        model_heatmaps[i:i+1].unsqueeze(0), 
                        size=(256, 256), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    # heatmap overlay
                    axes[row, col].imshow(image_np)
                    im = axes[row, col].imshow(heatmap_resized, alpha=0.6, cmap='hot', vmin=0, vmax=1)
                    axes[row, col].set_title(f"{model_name}\nKeypoint {i}")
                    axes[row, col].axis('off')
                    
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualization to sample_visualization.png")
        plt.close()
        
        break  # only show one sample

if __name__ == "__main__":
    visualize_sample() 