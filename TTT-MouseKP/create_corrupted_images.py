import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse

def add_gaussian_noise(image, std=240):
    """Add Gaussian noise to image"""
    img_array = np.array(image)
    noise = np.random.normal(0, std, img_array.shape)
    corrupted = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(corrupted)

def add_motion_blur(image, size=15):
    """Add motion blur to image using simple blur"""
    # Use stronger blur to simulate motion blur
    blurred = image.filter(ImageFilter.GaussianBlur(radius=8.0))
    return blurred

def add_defocus_blur(image, radius=12):
    """Add defocus blur"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_gamma_correction(image, gamma=0.1):
    """Apply gamma correction"""
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)


def add_snow(image, density=0.3):
    """Add snow effect"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # create snow mask with random seed for consistency
    np.random.seed(42)
    snow_mask = np.random.random((height, width)) < density
    
    # create larger snowflakes by dilating
    kernel_size = 3
    for i in range(-kernel_size//2, kernel_size//2 + 1):
        for j in range(-kernel_size//2, kernel_size//2 + 1):
            if 0 <= i < height and 0 <= j < width:
                shifted_mask = np.roll(np.roll(snow_mask, i, axis=0), j, axis=1)
                snow_mask = snow_mask | shifted_mask
    
    # apply snow
    if len(img_array.shape) == 3:
        for c in range(3):
            img_array[:, :, c] = np.where(snow_mask, 
                                        np.minimum(img_array[:, :, c] + 100, 255),
                                        img_array[:, :, c])
    else:
        img_array = np.where(snow_mask, 
                           np.minimum(img_array + 100, 255),
                           img_array)
    
    return Image.fromarray(img_array.astype(np.uint8))

def add_glass_blur(image, blur_radius=2, local_variance=0.5):
    """Add glass blur effect (frosted glass distortion)"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # create random displacement field for glass effect
    np.random.seed(42)  # For reproducibility
    
    # generate random displacement maps
    dx = np.random.normal(0, local_variance, (height, width))
    dy = np.random.normal(0, local_variance, (height, width))
    
    # apply Gaussian smoothing to displacement maps
    from scipy import ndimage
    dx = ndimage.gaussian_filter(dx, sigma=blur_radius)
    dy = ndimage.gaussian_filter(dy, sigma=blur_radius)
    
    # create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # apply displacement
    new_x = x_coords + dx
    new_y = y_coords + dy
    
    # clamp coordinates to image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    # sample from original image using bilinear interpolation
    if len(img_array.shape) == 3:
        result = np.zeros_like(img_array)
        for c in range(img_array.shape[2]):
            result[:, :, c] = ndimage.map_coordinates(
                img_array[:, :, c], [new_y, new_x], order=1, mode='reflect'
            )
    else:
        result = ndimage.map_coordinates(
            img_array, [new_y, new_x], order=1, mode='reflect'
        )
    
    return Image.fromarray(result.astype(np.uint8))

def add_shot_noise(image, noise_scale=40):
    """Add shot noise (Poisson noise)"""
    img_array = np.array(image).astype(np.float32)
    
    np.random.seed(42)
    
    scaled = img_array * noise_scale / 255.0
    
    # Apply Poisson noise
    noisy = np.random.poisson(scaled).astype(np.float32)
    
    shot_noise = (noisy * 255.0 / noise_scale) - img_array
    
    result = img_array + shot_noise * 0.3
    result = np.clip(result, 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))

def create_corrupted_dataset(input_dir, output_dir, corruption_type='noise'):
    """Create corrupted version of dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Creating {corruption_type} corrupted images...")
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        # Load image
        image = Image.open(input_path).convert('RGB')
        
        # Apply corruption with EXTREME parameters
        if corruption_type == 'noise':
            corrupted = add_gaussian_noise(image, std=120)
        elif corruption_type == 'blur':
            corrupted = add_defocus_blur(image, radius=12)
        elif corruption_type == 'motion_blur':
            corrupted = add_motion_blur(image, size=15)
        elif corruption_type == 'gamma_low':
            corrupted = apply_gamma_correction(image, gamma=0.1)
        elif corruption_type == 'gamma_high':
            corrupted = apply_gamma_correction(image, gamma=6.0)
        elif corruption_type == 'snow':
            corrupted = add_snow(image, density=0.3)
        elif corruption_type == 'glass_blur':
            corrupted = add_glass_blur(image, blur_radius=3, local_variance=1.0)
        elif corruption_type == 'shot_noise':
            corrupted = add_shot_noise(image, noise_scale=40)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        # save corrupted images
        corrupted.save(output_path)
        
    print(f"Saved {len(image_files)} corrupted images to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/mouse_dataset/val/images')
    parser.add_argument('--output_dir', default='data/mouse_dataset/val_corrupted')
    parser.add_argument('--corruption', default='noise', 
                       choices=['noise', 'blur', 'dark', 'motion_blur', 'snow', 'glass_blur', 'shot_noise'])
    args = parser.parse_args()
    
    create_corrupted_dataset(args.input_dir, args.output_dir, args.corruption)

if __name__ == '__main__':
    main() 