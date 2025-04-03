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

def improved_lane_detection(frame, roi_mask):
    """
    Improved method for detecting lane markings in frames
    
    Args:
        frame: Input image frame
        roi_mask: Region of interest mask
        
    Returns:
        Binary mask with lane markings
    """
    # Get dimensions
    height, width = frame.shape[:2]
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered_frame = cv2.bilateralFilter(frame, 7, 50, 50)
    
    # -------- HSV COLOR SPACE DETECTION --------
    hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
    
    # White lanes detection (fine-tuned)
    white_lower = np.array([0, 0, 210])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Orange/yellow lanes detection (fine-tuned)
    orange_lower = np.array([10, 100, 160])
    orange_upper = np.array([30, 255, 255])
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    
    # -------- HLS COLOR SPACE DETECTION --------
    hls = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HLS)
    
    # White in HLS
    white_lower_hls = np.array([0, 200, 0])
    white_upper_hls = np.array([180, 255, 255])
    white_mask_hls = cv2.inRange(hls, white_lower_hls, white_upper_hls)
    
    # Yellow/orange in HLS 
    yellow_lower_hls = np.array([15, 100, 50])
    yellow_upper_hls = np.array([35, 205, 255])
    yellow_mask_hls = cv2.inRange(hls, yellow_lower_hls, yellow_upper_hls)
    
    # Combine all color masks
    color_mask = cv2.bitwise_or(white_mask, orange_mask)
    hls_mask = cv2.bitwise_or(white_mask_hls, yellow_mask_hls)
    combined_color_mask = cv2.bitwise_or(color_mask, hls_mask)
    
    # Apply ROI
    combined_color_mask = cv2.bitwise_and(combined_color_mask, roi_mask)
    
    # Create a position-weighted mask (focus on bottom portion)
    position_weight = np.ones_like(combined_color_mask, dtype=np.float32)
    for y in range(height):
        weight = 0.3 + 0.7 * (y / height)
        position_weight[y, :] = weight
    
    # Apply the position weighting
    weighted_mask = cv2.multiply(combined_color_mask.astype(np.float32), position_weight)
    weighted_mask = weighted_mask.astype(np.uint8)
    
    # Apply small morphological close to connect nearby points
    kernel_close = np.ones((3, 3), np.uint8)
    processed_mask = cv2.morphologyEx(weighted_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find and filter connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
    
    filtered_mask = np.zeros_like(processed_mask)
    
    for i in range(1, num_labels):
        # Get component properties
        area = stats[i, cv2.CC_STAT_AREA]
        comp_width = stats[i, cv2.CC_STAT_WIDTH]
        comp_height = stats[i, cv2.CC_STAT_HEIGHT]
        x_pos = stats[i, cv2.CC_STAT_LEFT]
        y_pos = stats[i, cv2.CC_STAT_TOP]
        
        # Calculate component center and relative positions
        center_x = x_pos + comp_width/2
        rel_x_pos = center_x / width
        rel_y_pos = y_pos / height
        
        # Adaptive criteria based on position
        min_area = 10 + 70 * (1.0 - rel_y_pos)
        
        # Lane-like shape calculations
        aspect_ratio = comp_height / comp_width if comp_width > 0 else 999
        is_lane_shaped = ((aspect_ratio > 1.5 and comp_height > 25) or 
                         (comp_width < 80 and comp_height > 15) or
                         (area > 80 and comp_width < 50))
        
        is_valid_position = (rel_y_pos > 0.3)
        
        if area > min_area and is_lane_shaped and is_valid_position and area < 8000:
            component_mask = np.zeros_like(processed_mask)
            component_mask[labels == i] = 255
            filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
    
    # If nothing significant was detected, fall back to basic mask
    if cv2.countNonZero(filtered_mask) < 10:
        filtered_mask = processed_mask.copy()
    
    # Final cleanup for thin, clear lines
    kernel_enhance = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_enhance)
    
    # Very light dilation for slightly thicker lines where needed
    kernel_dilate = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)
    
    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(filtered_mask, (3, 3), 0)
    
    # Final threshold for clear lines
    _, final_mask = cv2.threshold(blurred_mask, 40, 255, cv2.THRESH_BINARY)
    
    # Apply ROI as final step
    final_mask = cv2.bitwise_and(final_mask, roi_mask)
    
    return final_mask

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
        
        # Use improved lane detection method
        final_mask = improved_lane_detection(frame, roi_mask)
        
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