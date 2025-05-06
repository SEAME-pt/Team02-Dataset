import cv2
import os
import numpy as np
import json

def load_jsonl(file_path):
    """
    Load a JSON Lines file where each line is a separate JSON object
    Returns a list of dictionaries
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//'):
                    data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading from {file_path}: {e}")
        return []

def create_lane_annotations(frames_folder, output_json_path):
    """
    Interactive tool to create TuSimple-format lane annotations
    
    Controls:
    - N: Next image
    - P: Previous image
    - E: Toggle point placement mode
    - A: Add current lane and start a new one
    - F: Finish lane annotation for current image and save to JSON
    - D: Delete last point placed
    - C: Clear current lane
    - R: Reset all lanes on current image
    - S: Save annotations to JSON file
    - Q: Quit
    """
    # Get sorted list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        print("No frames found!")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Current frame index
    current_idx = 0
    
    # Mode flags
    editing_mode = None  # None or "place_points"
    
    # Load existing annotations from JSON file if it exists
    annotations = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                annotations = load_jsonl(output_json_path)
            print(f"Successfully loaded {len(annotations)} annotations from {output_json_path}")
            
            # Print the first few annotations to verify
            for i, ann in enumerate(annotations[:3]):
                print(f"Loaded annotation for {ann.get('raw_file')}")
                
        except Exception as e:
            print(f"Error loading annotations from {output_json_path}: {e}")
            annotations = []
    else:
        print(f"No existing annotations file found at {output_json_path}")
        
    # Current image lanes
    current_lanes = []
    active_lane = []  # Points for the current lane being created
    
    # Mouse position tracking for preview
    mouse_x, mouse_y = -1, -1
    
    # Function to convert lanes to TuSimple format
    def lanes_to_tusimple(frame_path, lanes):
        if not lanes:
            print("WARNING: No lanes provided to lanes_to_tusimple")
            return None
                
        print(f"Processing {len(lanes)} lanes for TuSimple format")
        
        # Get image dimensions
        img = cv2.imread(frame_path)
        if img is None:
            print(f"ERROR: Could not read image {frame_path}")
            return None
                
        h, w = img.shape[:2]
        
        # First determine the overall y-range to sample
        all_y_coords = []
        for lane in lanes:
            if len(lane) >= 2:
                y_values = [p[1] for p in lane]
                all_y_coords.extend(y_values)
        
        if not all_y_coords:
            print("WARNING: No valid y-coordinates found in lanes")
            return None
        
        # Get min and max y across all points
        min_y = min(all_y_coords)
        max_y = max(all_y_coords)
        
        # Generate evenly spaced h_samples between min_y and max_y
        num_points = 50  # TuSimple format typically uses ~50 points
        h_samples = [int(y) for y in np.linspace(min_y, max_y, num_points)]
        print(f"Using {len(h_samples)} h_samples from y={min_y} to y={max_y}")
        
        # For each lane, create a polyline using original point order 
        # then sample at the required y-coordinates
        x_lanes = []
        for lane_idx, lane in enumerate(lanes):
            if len(lane) < 2:
                print(f"WARNING: Lane {lane_idx} with {len(lane)} points is too short, skipping")
                continue
            
            # Use points IN THE EXACT ORDER they were placed
            lane_points = lane  # No sorting
            lane_x = [p[0] for p in lane_points]
            lane_y = [p[1] for p in lane_points]
            
            print(f"Lane {lane_idx}: Processing {len(lane_points)} points in original placement order")
            
            # Create a polyline connecting the points in the exact order they were placed
            points = np.array(lane_points)
            
            # For each h_sample, find the closest point on the polyline
            lane_x_interp = []
            
            for y_target in h_samples:
                # Find if this y value is within the range of our manually placed points
                min_lane_y = min(lane_y)
                max_lane_y = max(lane_y)
                
                if min_lane_y <= y_target <= max_lane_y:
                    # Find the line segment that contains this y value
                    for i in range(len(lane_points) - 1):
                        y1 = lane_points[i][1]
                        y2 = lane_points[i+1][1]
                        
                        # Check if y_target is between y1 and y2 (inclusive)
                        if (y1 <= y_target <= y2) or (y2 <= y_target <= y1):
                            x1 = lane_points[i][0]
                            x2 = lane_points[i+1][0]
                            
                            # If y1 equals y2, use the midpoint
                            if y1 == y2:
                                x_interp = (x1 + x2) // 2
                            else:
                                # Linear interpolation
                                ratio = (y_target - y1) / (y2 - y1)
                                x_interp = int(x1 + ratio * (x2 - x1))
                            
                            lane_x_interp.append(x_interp)
                            break
                    else:
                        # If we didn't find a segment, use the nearest point
                        y_diffs = [abs(y - y_target) for y in lane_y]
                        nearest_idx = y_diffs.index(min(y_diffs))
                        lane_x_interp.append(lane_x[nearest_idx])
                else:
                    # Outside the range of our points
                    lane_x_interp.append(-2)
            
            x_lanes.append(lane_x_interp)
            valid_points = len([x for x in lane_x_interp if x != -2])
            print(f"Lane {lane_idx}: Generated {valid_points} valid points")
        
        if not x_lanes:
            print("WARNING: No valid lanes after processing")
            return None
        
        # Create TuSimple format annotation
        annotation = {
            "raw_file": os.path.basename(frame_path),
            "lanes": x_lanes,
            "h_samples": h_samples,
            "original_lanes": [lane.copy() for lane in lanes]
        }
        
        print(f"Successfully created annotation with {len(x_lanes)} lanes")
        return annotation
    
    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal active_lane, editing_mode
        nonlocal mouse_x, mouse_y
        
        # Update mouse position for preview
        mouse_x, mouse_y = x, y
        
        if editing_mode is None:
            return
        
        # Update display to show preview at current position
        if editing_mode == "place_points":
            update_display()
            
        # Place point
        if event == cv2.EVENT_LBUTTONDOWN and editing_mode == "place_points":
            active_lane.append((x, y))
            print(f"Placed point at ({x}, {y})")
            update_display()
    
    # Function to update the display
    def update_display():
        nonlocal current_frame, current_lanes, active_lane, editing_mode
        nonlocal mouse_x, mouse_y
        
        # Create a display image
        display_image = current_frame.copy()
        
        # Colors for existing lanes
        lane_colors = [
            (0, 255, 0),    # Green
            (0, 165, 255),  # Orange
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Magenta
        ]
        
        # Enhanced visualization in view mode (not editing)
        if editing_mode is None and current_lanes:
            # Create a semi-transparent overlay for better visibility
            overlay = display_image.copy()
            
            # Draw lanes with thicker lines and larger points
            for i, lane in enumerate(current_lanes):
                color = lane_colors[i % len(lane_colors)]
                
                # Draw thicker lane lines for better visibility
                for j in range(len(lane) - 1):
                    cv2.line(overlay, lane[j], lane[j+1], color, 6)
                
                # Draw larger points
                for point in lane:
                    cv2.circle(overlay, point, 8, color, -1)
                
                # Add lane number
                if lane:
                    cv2.putText(overlay, f"Lane {i+1}", 
                               (lane[0][0] + 10, lane[0][1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Apply semi-transparent overlay
            alpha = 0.7  # Transparency factor
            display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)
            
            # Add an indicator that we're in enhanced view mode
            cv2.putText(display_image, "ENHANCED VIEW MODE", (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            # Standard editing display - draw completed lanes
            for i, lane in enumerate(current_lanes):
                color = lane_colors[i % len(lane_colors)]
                
                # Draw points
                for point in lane:
                    cv2.circle(display_image, point, 5, color, -1)
                
                # Draw lines connecting points
                for j in range(len(lane) - 1):
                    cv2.line(display_image, lane[j], lane[j+1], color, 2)
                    
                # Label the lane
                if lane:
                    cv2.putText(display_image, f"Lane {i+1}", 
                               (lane[0][0] + 10, lane[0][1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw the active lane (being created)
            if active_lane:
                active_color = (255, 255, 0)  # Cyan
                
                # Draw points
                for point in active_lane:
                    cv2.circle(display_image, point, 5, active_color, -1)
                
                # Draw lines connecting points
                for j in range(len(active_lane) - 1):
                    cv2.line(display_image, active_lane[j], active_lane[j+1], active_color, 2)
                    
                # Label as active lane
                cv2.putText(display_image, f"Active Lane", 
                           (active_lane[0][0] + 10, active_lane[0][1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2)
            
            # Show preview point at current mouse position if placing points
            if editing_mode == "place_points" and mouse_x >= 0 and mouse_y >= 0:
                cv2.circle(display_image, (mouse_x, mouse_y), 5, (255, 255, 255), 2)
        
        # Show frame info
        frame_file = frame_files[current_idx]
        status_text = f"Frame: {current_idx+1}/{len(frame_files)} | {frame_file}"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show mode info
        mode_text = f"MODE: {'PLACING POINTS' if editing_mode == 'place_points' else 'VIEWING'}"
        cv2.putText(display_image, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show lane count
        lane_text = f"Lanes: {len(current_lanes)} completed + {1 if active_lane else 0} active"
        cv2.putText(display_image, lane_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show help info
        cv2.putText(display_image, "Press H for help", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Lane Annotation", display_image)
    
    # Create window and set mouse callback
    cv2.namedWindow("Lane Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Lane Annotation", mouse_callback)
    
    # Function to load a frame and its annotations
    def load_frame(idx):
        nonlocal current_lanes, active_lane, mouse_x, mouse_y
        
        # Get frame info
        frame_file = frame_files[idx]
        frame_path = os.path.join(frames_folder, frame_file)
        
        # Load the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"ERROR: Could not read image {frame_path}")
            return None, frame_file, frame_path
        
        # Check if we already have annotations for this frame
        existing_annotation = None
        print(f"Looking for annotation for {frame_file}")
        for ann in annotations:
            if ann.get("raw_file") == frame_file:
                existing_annotation = ann
                print(f"Found existing annotation for {frame_file}")
                break
        
        # Reset lane data when loading a new frame
        if existing_annotation:
            # First try to use original lanes if available
            if "original_lanes" in existing_annotation:
                # Convert the stored list format back to tuples
                current_lanes = []
                for lane in existing_annotation["original_lanes"]:
                    if isinstance(lane, list):
                        # Convert the points to tuples
                        current_lanes.append([(int(x), int(y)) for x, y in lane])
                    else:
                        current_lanes.append(lane)
                print(f"Loaded {len(current_lanes)} lanes from original points")
            else:
                # Fall back to reconstructing from TuSimple format
                h_samples = existing_annotation.get("h_samples", [])
                x_lanes = existing_annotation.get("lanes", [])
                
                new_lanes = []
                for lane_x in x_lanes:
                    # Convert to list of points, filtering out missing values (-2)
                    points = [(int(x), int(y)) for x, y in zip(lane_x, h_samples) if x != -2]
                    if points:
                        new_lanes.append(points)
                        print(f"Reconstructed lane with {len(points)} points")
                
                current_lanes = new_lanes
                print(f"Loaded {len(current_lanes)} lanes from TuSimple format")
        else:
            current_lanes = []
            print(f"No existing annotation found for {frame_file}")
        
        active_lane = []
        
        # Reset mouse position
        mouse_x, mouse_y = -1, -1
        
        return frame, frame_file, frame_path
    
    # Load initial frame
    current_frame, frame_file, frame_path = load_frame(current_idx)
    
    # Main loop
    while True:
        if current_idx < 0:
            current_idx = 0
        if current_idx >= len(frame_files):
            current_idx = len(frame_files) - 1
            
        # Update the display with the current frame and lanes
        update_display()
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit
            save_confirm = input("Save annotations before quitting? (y/n): ")
            if save_confirm.lower() == 'y':
                with open(output_json_path, 'w') as f:
                    # Write each annotation as a separate line (JSONL format)
                    for ann in annotations:
                        if ann:  # Skip any None values
                            f.write(json.dumps(ann) + '\n')
                print(f"Saved {len(annotations)} annotations to {output_json_path} in JSON Lines format")
            break
            
        elif key == ord('n'):  # Next image
            # Save current image annotation if lanes exist
            if active_lane:
                print("You have an active lane. Add it (A) or clear it (C) before continuing.")
            elif current_lanes:
                # Add annotation for current frame
                annotation = lanes_to_tusimple(frame_path, current_lanes)
                
                # Update or add annotation
                found = False
                for i, ann in enumerate(annotations):
                    if ann.get("raw_file") == frame_file:
                        annotations[i] = annotation
                        found = True
                        break
                
                if not found and annotation:
                    annotations.append(annotation)
                
                print(f"Saved annotation for {frame_file}")
                current_idx += 1
                current_frame, frame_file, frame_path = load_frame(current_idx)
            else:
                current_idx += 1
                current_frame, frame_file, frame_path = load_frame(current_idx)
            
        elif key == ord('p'):  # Previous image
            if active_lane:
                print("You have an active lane. Add it (A) or clear it (C) before continuing.")
            else:
                current_idx -= 1
                current_frame, frame_file, frame_path = load_frame(current_idx)
            
        elif key == ord('e'):  # Toggle point placement mode
            editing_mode = "place_points" if editing_mode != "place_points" else None
            print(f"Point placement mode {'activated' if editing_mode == 'place_points' else 'deactivated'}")
            update_display()
            
        elif key == ord('a'):  # Add current lane and start a new one
            if active_lane and len(active_lane) >= 2:
                current_lanes.append(active_lane.copy())  # Create a copy of the lane before adding
                print(f"Added lane with {len(active_lane)} points, current_lanes now has {len(current_lanes)} lanes")
                active_lane = []
                update_display()
            else:
                print("Need at least 2 points to create a lane")
                
        elif key == ord('f'):  # Finish annotation for current image
            print(f"Before finishing: current_lanes has {len(current_lanes)} lanes")
            if active_lane:
                print("You have an active lane. Add it (A) or clear it (C) before finishing.")
            else:
                # Create annotation for current frame
                print(f"Creating annotation with {len(current_lanes)} lanes")
                annotation = lanes_to_tusimple(frame_path, current_lanes)
                
                # Debug output
                print(f"Generated annotation: {annotation}")
                
                # Update or add annotation
                found = False
                for i, ann in enumerate(annotations):
                    if ann.get("raw_file") == frame_file:
                        annotations[i] = annotation
                        found = True
                        print(f"Updated existing annotation for {frame_file}")
                        break
                
                if not found and annotation:
                    annotations.append(annotation)
                    print(f"Added new annotation for {frame_file}")
                    print(f"Now have {len(annotations)} total annotations")
                elif not annotation:
                    print("ERROR: Failed to generate valid annotation")
                
                print(f"Finished annotation for {frame_file}")
                
        elif key == ord('d'):  # Delete last point
            if active_lane:
                active_lane.pop()
                print("Removed last point")
                update_display()
                
        elif key == ord('c'):  # Clear current lane
            active_lane = []
            print("Cleared active lane")
            update_display()
            
        elif key == ord('r'):  # Reset all lanes for current image
            active_lane = []
            current_lanes = []
            
            # Remove any existing annotation
            for i, ann in enumerate(annotations):
                if ann.get("raw_file") == frame_file:
                    annotations.pop(i)
                    break
                    
            print("Reset all lanes for current image")
            update_display()
            
        elif key == ord('s'):  # Save all annotations
            with open(output_json_path, 'w') as f:
                # Write each annotation as a separate line (JSONL format)
                for ann in annotations:
                    if ann:  # Skip any None values
                        f.write(json.dumps(ann) + '\n')
            print(f"Saved {len(annotations)} annotations to {output_json_path} in JSON Lines format")

            
        elif key == ord('h'):  # Help
            print("\nKEYBOARD SHORTCUTS:")
            print("N: Next image")
            print("P: Previous image")
            print("E: Toggle point placement mode")
            print("A: Add current lane and start a new one")
            print("F: Finish lane annotation for current image and save to JSON")
            print("D: Delete last point placed")
            print("C: Clear current lane")
            print("R: Reset all lanes on current image")
            print("S: Save annotations to JSON file")
            print("Q: Quit")
            print("\nMOUSE CONTROLS:")
            print("Left click: Place a point (in point mode)")
    
    cv2.destroyAllWindows()
    print("Annotation completed")

if __name__ == "__main__":
    frames_folder = "frames"
    output_json_path = "lane_annotations.json"
    
    create_lane_annotations(frames_folder, output_json_path)