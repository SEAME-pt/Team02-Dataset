import cv2
import os
import numpy as np
import json
import time

def visualize_frames_and_annotations(frames_folder, annotations_path, delay=100, full_screen=False):
    """
    Display frames and their corresponding lane annotations side by side as a video
    Press ESC to exit at any time
    
    Args:
        frames_folder: Folder containing extracted frames
        annotations_path: Path to TuSimple format JSON annotations file
        delay: Delay between frames in milliseconds (controls playback speed)
        full_screen: If True, shows only the overlay view full screen instead of side-by-side
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Create a lookup dictionary for easy access to annotations by filename
    annotations_dict = {}
    for ann in annotations:
        annotations_dict[ann.get("raw_file")] = ann
    
    # Get sorted list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    print(f"Found {len(frame_files)} frames to visualize")
    print(f"Loaded annotations for {len(annotations_dict)} frames")
    print("Displaying video. Press ESC to exit. Press 'F' to toggle full screen mode.")
    
    # Create a window that can be resized
    cv2.namedWindow("Lane Visualization", cv2.WINDOW_NORMAL)
    
    for i, frame_file in enumerate(frame_files):
        # Construct path
        frame_path = os.path.join(frames_folder, frame_file)
        
        # Read image
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        # Create lane overlay
        overlay = frame.copy()
        
        # Check if we have annotation for this frame
        if frame_file in annotations_dict:
            annotation = annotations_dict[frame_file]
            
            # Get lane points from annotation
            lanes = annotation.get("lanes", [])
            h_samples = annotation.get("h_samples", [])
            
            # Different colors for each lane
            lane_colors = [
                (0, 255, 0),    # Green
                (0, 165, 255),  # Orange
                (0, 0, 255),    # Red
                (255, 0, 0),    # Blue
                (255, 0, 255),  # Magenta
            ]
            
            # Draw each lane
            for lane_idx, lane_x in enumerate(lanes):
                color = lane_colors[lane_idx % len(lane_colors)]
                
                # Create points for the lane (filter out -2 values which mean no point)
                points = []
                for x, y in zip(lane_x, h_samples):
                    if x != -2:  # -2 means no lane point at this y-coordinate
                        points.append((int(x), int(y)))
                
                # Debug message
                print(f"Lane {lane_idx}: Found {len(points)} valid points out of {len(lane_x)} total")
                
                # IMPROVED APPROACH: Draw connected lines between all valid points
                for i in range(len(points) - 1):
                    # Draw a line between consecutive valid points
                    cv2.line(overlay, points[i], points[i+1], color, 8)
                    
                # Draw points on top of lines for better visibility
                for point in points:
                    cv2.circle(overlay, point, 10, color, -1)
                    
                # Add lane number for identification
                if points:
                    cv2.putText(overlay, f"Lane {lane_idx+1}", 
                              points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Add annotation info
            cv2.putText(overlay, f"Lanes: {len(lanes)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(overlay, "No annotation", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Choose between side-by-side or full screen display
        if full_screen:
            display_frame = overlay  # Just show the overlay
        else:
            display_frame = np.hstack((frame, overlay))  # Side by side
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {i+1}/{len(frame_files)} - {frame_file}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Lane Visualization", display_frame)
        
        time.sleep(1)
        
        # Wait for a short delay (creates video effect) or check for key press
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key
            break
        elif key == ord('f'):  # Toggle full screen mode
            full_screen = not full_screen
            print(f"Full screen mode: {'ON' if full_screen else 'OFF'}")
    
    cv2.destroyAllWindows()
    print("Visualization completed")

if __name__ == "__main__":
    frames_folder = "frames"
    annotations_path = "lane_annotations.json"
    
    # Adjust the delay value to control playback speed (smaller = faster)
    # 33ms = ~30fps, 100ms = 10fps, 500ms = 2fps
    visualize_frames_and_annotations(frames_folder, annotations_path, delay=100, full_screen=False)