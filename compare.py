import cv2
import os
import numpy as np

def visualize_frames_and_masks(frames_folder, masks_folder, delay=100):
    """
    Display frames and their corresponding masks side by side as a video
    Press ESC to exit at any time
    
    Args:
        frames_folder: Folder containing extracted frames
        masks_folder: Folder containing generated masks
        delay: Delay between frames in milliseconds (controls playback speed)
    """
    # Get sorted list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    print(f"Found {len(frame_files)} frames to visualize")
    print("Displaying video. Press ESC to exit.")
    
    # Create a window that can be resized
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    
    for i, frame_file in enumerate(frame_files):
        # Construct paths
        frame_path = os.path.join(frames_folder, frame_file)
        mask_path = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {frame_file}, skipping")
            continue
        
        # Read images
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if frame is None or mask is None:
            print(f"Warning: Could not read frame or mask for {frame_file}")
            continue
        
        # Resize if needed
        if frame.shape[0] != mask.shape[0] or frame.shape[1] != mask.shape[1]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Create overlay image
        overlay = frame.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red for lane pixels
        
        # Combine for side-by-side view
        frame_with_overlay = np.hstack((frame, overlay))
        
        # Add frame counter
        cv2.putText(frame_with_overlay, f"Frame: {i+1}/{len(frame_files)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the concatenated image
        cv2.imshow("Lane Detection", frame_with_overlay)
        
        # Wait for a short delay (creates video effect) or check for ESC key
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    print("Visualization completed")

if __name__ == "__main__":
    frames_folder = "frames"
    masks_folder = "masks"
    
    # Adjust the delay value to control playback speed (smaller = faster)
    # 33ms = ~30fps, 100ms = 10fps, 500ms = 2fps
    visualize_frames_and_masks(frames_folder, masks_folder, delay=100)