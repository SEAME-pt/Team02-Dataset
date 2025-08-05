import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_unique_frames(video_path, output_folder, similarity_threshold=0.80, skip_frames=1):
    """
    Extract only frames that are significantly different from previously saved frames
    
    Args:
        video_path: Path to the video file
        output_folder: Path to the output folder
        similarity_threshold: Threshold for frame similarity (0-1)
            Higher values = more similar, fewer frames saved
            Lower values = less similar, more frames saved
        skip_frames: Process only every Nth frame for efficiency (default: 1 = check every frame)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, duration: {duration:.2f} seconds")
    print(f"Settings: similarity threshold = {similarity_threshold}, skip frames = {skip_frames}")
    
    # Initialize variables
    frame_number = 0
    saved_count = 8000
    prev_frame_gray = None
    last_saved_frame_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for efficiency (check only every Nth frame)
        if skip_frames > 1 and frame_number % skip_frames != 0:
            frame_number += 1
            continue
            
        # Convert to grayscale for comparison
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame is always saved
        if last_saved_frame_gray is None:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            last_saved_frame_gray = frame_gray
            print(f"Saved frame 0 (first frame)")
        else:
            # Compare with the last saved frame
            score, _ = ssim(last_saved_frame_gray, frame_gray, full=True)
            
            # If frames are different enough, save the current frame
            if score < similarity_threshold:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                last_saved_frame_gray = frame_gray
                
                # Print progress
                print(f"Saved frame {saved_count-1} (original #{frame_number}, similarity: {score:.4f})")
        
        frame_number += 1
        
        # Optional: show progress
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames, saved {saved_count} unique frames...")
    
    cap.release()
    print(f"Finished! Extracted {saved_count} unique frames from {frame_count} total frames")
    print(f"Achieved {saved_count/frame_count*100:.2f}% compression ratio")

if __name__ == "__main__":
    video_path = "assets/video.mp4"
    output_folder = "frames"
    
    extract_unique_frames(
        video_path, 
        output_folder, 
        similarity_threshold=1,  # Lowered from 0.95 to 0.80 for much more distinct frames
        skip_frames=5             # Check every 5th frame for efficiency
    )
    