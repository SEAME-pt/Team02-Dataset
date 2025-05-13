import os
import cv2
import torch
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SEAMERoadDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, width=256, height=128, is_train=True):
        """
        Dataset for road segmentation using polygon annotations
        
        Args:
            img_dir: Directory with source images
            annotation_file: Path to JSONL file with road annotations
            width, height: Target dimensions for resizing
            is_train: Whether to use training augmentations
        """
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        
        # Load annotations
        self.annotations = []
        print(f"Loading road annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.annotations.append(json.loads(line))
        
        print(f"Loaded {len(self.annotations)} road annotations")
        
        # Filter to only include annotations with available images
        self.annotations = [ann for ann in self.annotations if 
                           os.path.exists(os.path.join(img_dir, ann['raw_file']))]
        print(f"Found {len(self.annotations)} annotations with matching images")
        
        # Set up transformations
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get annotation
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation['raw_file'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from polygons
        mask = np.zeros((annotation['image_height'], annotation['image_width']), dtype=np.uint8)
        
        # Fill each polygon with class 1 (road)
        for polygon in annotation['polygons']:
            # Convert polygon to numpy array format for cv2.fillPoly
            points = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return image, mask

    def visualize(self, idx):
        """
        Visualize an image and its corresponding mask
        """
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation['raw_file'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask
        mask = np.zeros((annotation['image_height'], annotation['image_width']), dtype=np.uint8)
        for polygon in annotation['polygons']:
            points = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
        
        # Create colored overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = [0, 255, 0]  # Green for road
        
        # Blend image and mask
        alpha = 0.5
        blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        # Return visualization
        return blended
    
def main():
    # Hardcoded paths - change these to match your setup
    img_dir = 'frames'
    annotation_file = 'road_annotations.json'
    
    # Load the dataset
    dataset = SEAMERoadDataset(
        img_dir=img_dir,
        annotation_file=annotation_file,
        width=512,  # Using larger size for visualization
        height=256,
        is_train=False  # No augmentations for visualization
    )
    
    # Check if we have any annotations
    if len(dataset) == 0:
        print("No annotations found. Please check your paths.")
        return
    
    # Variables for navigation
    current_idx = 0
    
    print("\n--- Road Annotation Viewer ---")
    print(f"Dataset contains {len(dataset)} annotated images")
    print("Controls:")
    print("  Next image: Right arrow or 'n'")
    print("  Previous image: Left arrow or 'p'")
    print("  Quit: 'q' or ESC")
    
    while True:
        # Get visualization for current index
        vis = dataset.visualize(current_idx)
        
        # Convert from RGB to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Get annotation info
        annotation = dataset.annotations[current_idx]
        filename = annotation['raw_file']
        num_polygons = len(annotation['polygons'])
        
        # Add text with file information
        info_text = f"Image {current_idx+1}/{len(dataset)}: {filename} ({num_polygons} polygons)"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Road Annotations", vis)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Handle key press
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('n') or key == 83:  # n or right arrow
            current_idx = min(current_idx + 1, len(dataset) - 1)
        elif key == ord('p') or key == 81:  # p or left arrow
            current_idx = max(current_idx - 1, 0)
    
    cv2.destroyAllWindows()
    print("Visualization complete!")

if __name__ == "__main__":
    main()