import os
import cv2
import torch
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SEAMESegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, width=256, height=128, is_train=True):
        """
        Dataset for multi-class segmentation using polygon annotations
        
        Args:
            img_dir: Directory with source images
            annotation_file: Path to JSONL file with annotations
            width, height: Target dimensions for resizing
            is_train: Whether to use training augmentations
        """
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        
        # Class definitions
        self.class_map = {
            "background": 0,
            "drivable_area": 1,
            "car": 2
        }
        
        self.class_colors = {
            0: [0, 0, 0],         # Background - Black
            1: [0, 255, 0],       # Road - Green
            2: [255, 0, 0]        # Car - Blue
        }
        
        # Load annotations
        self.annotations = []
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.annotations.append(json.loads(line))
        
        print(f"Loaded {len(self.annotations)} annotations")
        
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
        
        # Check if we have the new multi-class format or old format
        if 'annotations' in annotation:
            # New format with multiple classes
            for class_ann in annotation['annotations']:
                class_type = class_ann['type']
                if class_type in self.class_map:
                    class_id = self.class_map[class_type]
                    for polygon in class_ann['polygons']:
                        # Convert polygon to numpy array format for cv2.fillPoly
                        points = np.array([polygon], dtype=np.int32)
                        cv2.fillPoly(mask, points, class_id)
        else:
            # Old format with just road polygons
            for polygon in annotation.get('polygons', []):
                # Convert polygon to numpy array format for cv2.fillPoly
                points = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(mask, points, self.class_map["drivable_area"])
        
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
        
        # Track classes in this image
        classes_present = set()
        
        # Check if we have the new multi-class format or old format
        if 'annotations' in annotation:
            # New format with multiple classes
            for class_ann in annotation['annotations']:
                class_type = class_ann['type']
                if class_type in self.class_map:
                    class_id = self.class_map[class_type]
                    classes_present.add(class_type)
                    for polygon in class_ann['polygons']:
                        points = np.array([polygon], dtype=np.int32)
                        cv2.fillPoly(mask, points, class_id)
        else:
            # Old format with just road polygons
            classes_present.add("drivable_area")
            for polygon in annotation.get('polygons', []):
                points = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(mask, points, self.class_map["drivable_area"])
        
        # Create colored overlay
        colored_mask = np.zeros_like(image)
        
        # Apply colors for each class
        for class_id, color in self.class_colors.items():
            if class_id > 0:  # Skip background
                colored_mask[mask == class_id] = color
        
        # Blend image and mask
        alpha = 0.5
        blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        # Return visualization and classes
        return blended, classes_present
    
def main():
    # Hardcoded paths - change these to match your setup
    img_dir = 'frames'
    annotation_file = 'road_annotations.json'
    
    # Load the dataset
    dataset = SEAMESegmentationDataset(
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
    
    print("\n--- Segmentation Annotation Viewer ---")
    print(f"Dataset contains {len(dataset)} annotated images")
    print("Controls:")
    print("  Next image: Right arrow or 'n'")
    print("  Previous image: Left arrow or 'p'")
    print("  Quit: 'q' or ESC")
    
    while True:
        # Get visualization for current index
        vis, classes_present = dataset.visualize(current_idx)
        
        # Convert from RGB to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Get annotation info
        annotation = dataset.annotations[current_idx]
        filename = annotation['raw_file']
        
        # Count polygons by class
        polygon_counts = {}
        if 'annotations' in annotation:
            for class_ann in annotation['annotations']:
                class_type = class_ann['type']
                polygon_counts[class_type] = len(class_ann['polygons'])
        else:
            polygon_counts['drivable_area'] = len(annotation.get('polygons', []))
        
        # Add text with file information
        info_text = f"Image {current_idx+1}/{len(dataset)}: {filename}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Add class information
        y_pos = 60
        for class_name, count in polygon_counts.items():
            class_id = dataset.class_map.get(class_name, 0)
            color = dataset.class_colors.get(class_id, (255, 255, 255))
            # Convert RGB to BGR for OpenCV
            cv_color = (color[2], color[1], color[0])
            
            class_text = f"{class_name.replace('_', ' ').title()}: {count} polygons"
            cv2.putText(vis, class_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, cv_color, 2)
            y_pos += 30
        
        # Display the image
        cv2.imshow("Segmentation Annotations", vis)
        
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