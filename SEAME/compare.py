import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from SEAMEDataset import SEAMEDataset

def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def visualize_sample(image, binary_mask, instance_masks, n_lanes):
    """Visualize a single sample from the dataset"""
    # Convert image tensor to numpy array
    img = denormalize_image(image)
    
    # Get binary mask (channel 1 contains the lane pixels)
    bin_mask = binary_mask[1].numpy().astype(np.uint8) * 255
    
    # Create binary mask overlay
    bin_overlay = img.copy()
    bin_overlay[bin_mask > 0] = [0, 255, 0]  # Green color for lanes
    
    # Blend with original image
    bin_result = cv2.addWeighted(img, 0.7, bin_overlay, 0.3, 0)
    
    # Create instance mask overlay with different colors
    colors = [
        [0, 255, 0],    # Green
        [0, 165, 255],  # Orange
        [0, 0, 255],    # Red
        [255, 0, 0],    # Blue
        [255, 0, 255],  # Magenta
    ]
    
    ins_overlay = img.copy()
    for i in range(n_lanes):
        lane_mask = instance_masks[i].numpy().astype(np.uint8) * 255
        for c in range(3):
            ins_overlay[:, :, c][lane_mask > 0] = colors[i % len(colors)][c]
    
    # Blend with original image
    ins_result = cv2.addWeighted(img, 0.7, ins_overlay, 0.3, 0)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(bin_result)
    plt.title("Binary Lane Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ins_result)
    plt.title(f"Instance Segmentation ({n_lanes} lanes)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Define paths for your dataset
    json_path = "lane_annotations.json"
    img_dir = "frames"  # Adjust this path if needed
    
    # Create the dataset
    dataset = SEAMEDataset(
        json_paths=[json_path],
        img_dir=img_dir,
        width=512,
        height=256,
        is_train=False
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Visualize a few samples
    num_samples = 5
    for i in range(min(num_samples, len(dataset))):
        print(f"Visualizing sample {i+1}/{num_samples}")
        
        # Get a sample
        image, binary_mask, instance_masks, n_lanes = dataset[i]
        
        # Visualize it
        visualize_sample(image, binary_mask, instance_masks, n_lanes)
    
if __name__ == "__main__":
    main()