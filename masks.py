import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def filter_unique_frames(frames_folder, similarity_threshold=0.80):
    """
    Scan a folder of frames and return only unique frames based on similarity threshold
    
    Args:
        frames_folder: Path to the folder containing all frames
        similarity_threshold: Threshold for determining uniqueness (0-1)
    
    Returns:
        List of filenames of unique frames
    """
    # Get list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    print(f"Found {len(frame_files)} total frames, checking for unique ones...")
    
    unique_frames = []
    last_unique_frame_gray = None
    
    for frame_file in tqdm(frame_files):
        # Read the frame
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Convert to grayscale for comparison
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame is always unique
        if last_unique_frame_gray is None:
            unique_frames.append(frame_file)
            last_unique_frame_gray = frame_gray
        else:
            # Compare with the last unique frame
            score, _ = ssim(last_unique_frame_gray, frame_gray, full=True)
            
            # If frames are different enough, add to unique frames
            if score < similarity_threshold:
                unique_frames.append(frame_file)
                last_unique_frame_gray = frame_gray
    
    print(f"Identified {len(unique_frames)} unique frames out of {len(frame_files)} total frames")
    return unique_frames
def detect_lanes_and_create_masks(frames_folder, masks_folder, unique_only=True, similarity_threshold=0.80):
    """
    Process frames to detect lanes and create binary masks for training
    
    Args:
        frames_folder: Path to the folder containing all frames
        masks_folder: Path to the folder where masks will be saved
        unique_only: If True, only process unique frames based on similarity
        similarity_threshold: Threshold for determining uniqueness when unique_only=True
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
        print(f"Created output folder: {masks_folder}")
    
    # Determine which frames to process
    if unique_only:
        frame_files = filter_unique_frames(frames_folder, similarity_threshold)
    else:
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    print(f"Processing {len(frame_files)} frames to create masks")
    
    for frame_file in tqdm(frame_files):
        # Read the frame
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Get dimensions
        height, width = frame.shape[:2]
        
        # Create a ROI mask (exclude top portion of the image)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        # Fill only the bottom portion of the image - adjust the 0.25 as needed
        roi_mask[int(height*0.25):height, 0:width] = 255
        
        # ---------- COLOR-BASED DETECTION (FOCUSED ON WHITE & ORANGE) ----------
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for orange lane (replaces yellow)
        orange_lower = np.array([5, 80, 150])
        orange_upper = np.array([25, 255, 255])
        
        # Define range for white lane (more strict)
        white_lower = np.array([0, 0, 180])  # Higher value threshold for cleaner detection
        white_upper = np.array([180, 40, 255])
        
        # Create binary masks for orange and white
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine orange and white masks
        color_mask = cv2.bitwise_or(orange_mask, white_mask)
        
        # Apply ROI to color mask
        color_mask = cv2.bitwise_and(color_mask, roi_mask)
        
        # Apply morphological operations to clean up the mask
        # Use smaller kernels for thinner lines
        kernel_close = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Create a position-weighted mask to reduce noise in upper portion
        position_weight = np.ones_like(color_mask, dtype=np.float32)
        for y in range(height):
            # Weight increases as we move down the image (1.0 at bottom, 0.3 at top)
            weight = 0.3 + 0.7 * (y / height)
            position_weight[y, :] = weight
            
        # Apply the position weighting - upper part becomes darker
        weighted_mask = cv2.multiply(color_mask.astype(np.float32), position_weight)
        weighted_mask = weighted_mask.astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(weighted_mask, connectivity=8)
        
        # Filter components with position-adaptive criteria
        filtered_mask = np.zeros_like(color_mask)
        
        for i in range(1, num_labels):  # Skip background (0)
            # Get component properties
            area = stats[i, cv2.CC_STAT_AREA]
            comp_width = stats[i, cv2.CC_STAT_WIDTH]
            comp_height = stats[i, cv2.CC_STAT_HEIGHT]
            y_pos = stats[i, cv2.CC_STAT_TOP]  # Y position from top
            
            # Calculate relative position in image (0 = top, 1 = bottom)
            rel_position = y_pos / height
            
            # Adaptive criteria based on position
            min_area = 15 + 80 * (1.0 - rel_position)  # Lower minimum area for thinner lines
            shape_factor = 1.1 + 0.6 * (1.0 - rel_position)  # Less strict shape requirements
            
            # Apply adaptive criteria - adjusted for thinner lines
            # Allow smaller components and less strict shape requirements
            if area > min_area and (comp_height > comp_width * shape_factor or comp_width > 50) and area < 10000:
                # Create mask for this component
                component_mask = np.zeros_like(color_mask)
                component_mask[labels == i] = 255
                
                # Add to filtered mask
                filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
        
        # If nothing was detected, fall back to original weighted mask
        if cv2.countNonZero(filtered_mask) < 10:
            filtered_mask = weighted_mask.copy()
        
        # Enhance lane lines with a smaller kernel for thinner lines
        kernel_enhance = np.ones((3, 3), np.uint8)  # Smaller kernel
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_enhance)
        
        # Less dilation for thinner lines
        kernel_dilate = np.ones((3, 3), np.uint8)  # Smaller kernel
        filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)  # Fewer iterations
        
        # Apply Gaussian blur with smaller kernel
        filtered_mask = cv2.GaussianBlur(filtered_mask, (3, 3), 0)
        
        # Binarize with higher threshold for thinner lines
        _, final_mask = cv2.threshold(filtered_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Apply ROI mask as a final step to ensure top portion is excluded
        final_mask = cv2.bitwise_and(final_mask, roi_mask)
        
        # Save the lane mask
        mask_filename = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        cv2.imwrite(mask_filename, final_mask)
    
    print(f"Finished! Created masks for {len(frame_files)} frames")

if __name__ == "__main__":
    frames_folder = "frames" 
    masks_folder = "masks"
    
    # Process only unique frames, with the same similarity threshold used in frames.py
    detect_lanes_and_create_masks(
        frames_folder, 
        masks_folder,
        unique_only=True,
        similarity_threshold=0.72  # Use the same threshold as in frames.py
    )
    
    # Alternatively, to process all frames:
    # detect_lanes_and_create_masks(frames_folder, masks_folder, unique_only=False)