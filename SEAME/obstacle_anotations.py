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

def create_road_annotations(frames_folder, output_json_path):
    """
    Interactive tool to create road and car area annotations
    
    Controls:
    - N: Next image
    - P: Previous image
    - E: Toggle polygon point placement mode
    - A: Add current polygon and start a new one
    - F: Finish annotation for current image and save to JSON
    - D: Delete last point placed
    - C: Clear current polygon
    - R: Reset all polygons on current image
    - S: Save annotations to JSON file
    - T: Toggle between classes (road/car)
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
    
    # Define available classes and colors
    available_classes = ["drivable_area", "car",  "sign", "lights"]
    current_class = "drivable_area"  # Default class
    
    class_colors = {
        "drivable_area": [(0, 255, 0)], 
        "car": [(255, 0, 0)], 
        "sign": [(255, 0, 255)],
        "lights": [(0, 165, 255)]
    }
    
    # Load existing annotations from JSON file if it exists
    annotations = []
    if os.path.exists(output_json_path):
        try:
            annotations = load_jsonl(output_json_path)
            print(f"Successfully loaded {len(annotations)} annotations from {output_json_path}")
        except Exception as e:
            print(f"Error loading annotations from {output_json_path}: {e}")
            annotations = []
    
    # Current image annotations (will be a list of dicts with points and class)
    current_polygons = []
    active_polygon = {"points": [], "class": current_class}  # Points for the current polygon being created
    
    # Mouse position tracking for preview
    mouse_x, mouse_y = -1, -1
    
    # Function to create annotation in suitable format
    def create_annotation(frame_path, polygons):
        if not polygons:
            print("WARNING: No polygons provided")
            return None
            
        print(f"Processing {len(polygons)} polygons for annotation")
        
        # Get image dimensions
        img = cv2.imread(frame_path)
        if img is None:
            print(f"ERROR: Could not read image {frame_path}")
            return None
                
        h, w = img.shape[:2]
        
        # Group polygons by class
        polygons_by_class = {}
        for polygon in polygons:
            obj_class = polygon["class"]
            if obj_class not in polygons_by_class:
                polygons_by_class[obj_class] = []
            polygons_by_class[obj_class].append(polygon["points"])
        
        # Create annotation
        annotation = {
            "raw_file": os.path.basename(frame_path),
            "image_height": h,
            "image_width": w,
            "annotations": []
        }
        
        # Add each class's polygons
        for obj_class, class_polygons in polygons_by_class.items():
            annotation["annotations"].append({
                "type": obj_class,
                "polygons": [polygon.copy() for polygon in class_polygons]
            })
        
        print(f"Successfully created annotation with data for {len(polygons_by_class)} classes")
        return annotation
    
    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal active_polygon, editing_mode
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
            active_polygon["points"].append((x, y))
            print(f"Placed point at ({x}, {y}) for class: {active_polygon['class']}")
            update_display()
    
    # Function to update the display
    def update_display():
        nonlocal current_frame, current_polygons, active_polygon, editing_mode
        nonlocal mouse_x, mouse_y, current_class
        
        # Create a display image
        display_image = current_frame.copy()
        
        # Create overlay for filled polygons
        overlay = display_image.copy()
        
        # Draw completed polygons as filled areas with transparency
        for i, polygon in enumerate(current_polygons):
            points = polygon["points"]
            obj_class = polygon["class"]
            
            if len(points) >= 3:  # Need at least 3 points for a polygon
                # Get color based on class
                color_list = class_colors[obj_class]
                color = color_list[i % len(color_list)]
                
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Draw filled polygon with transparency
                cv2.fillPoly(overlay, [pts], color)
                
                # Draw polygon outline
                for j in range(len(points)):
                    next_j = (j + 1) % len(points)
                    cv2.line(display_image, points[j], points[next_j], color, 2)
                    cv2.circle(display_image, points[j], 5, color, -1)
                
                # Add polygon class and number
                label = f"{obj_class.replace('_', ' ').title()} {i+1}"
                #cv2.putText(display_image, label, 
                #           (points[0][0] + 10, points[0][1]), 
                 #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Apply the overlay with transparency
        alpha = 0.4  # Transparency factor
        display_image = cv2.addWeighted(overlay, alpha, display_image, 1.0, 0)
            
        # Draw the active polygon (being created)
        if active_polygon["points"]:
            # Get color based on active class
            active_color = class_colors[active_polygon["class"]][0]
            
            # Draw points
            for point in active_polygon["points"]:
                cv2.circle(display_image, point, 5, active_color, -1)
            
            # Draw lines connecting points
            for j in range(len(active_polygon["points"])):
                next_j = (j + 1) % len(active_polygon["points"]) if editing_mode == "place_points" else j + 1
                if next_j < len(active_polygon["points"]):
                    cv2.line(display_image, active_polygon["points"][j], active_polygon["points"][next_j], active_color, 2)
            
            # Draw line from last point to mouse position when in point placement mode
            if editing_mode == "place_points" and active_polygon["points"] and mouse_x >= 0 and mouse_y >= 0:
                cv2.line(display_image, active_polygon["points"][-1], (mouse_x, mouse_y), active_color, 2)
                
                # Add preview for closing polygon if at least 3 points
                if len(active_polygon["points"]) >= 3:
                    cv2.line(display_image, (mouse_x, mouse_y), active_polygon["points"][0], active_color, 2)
                
            # Label as active polygon
            #cv2.putText(display_image, f"Active {active_polygon['class'].replace('_', ' ').title()}", 
                      # (active_polygon["points"][0][0] + 10, active_polygon["points"][0][1]), 
                      # cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2)
        
        # Show preview point at current mouse position if placing points
        if editing_mode == "place_points" and mouse_x >= 0 and mouse_y >= 0:
            cv2.circle(display_image, (mouse_x, mouse_y), 5, (255, 255, 255), 2)
        
        # Show frame info
        frame_file = frame_files[current_idx]
        # status_text = f"Frame: {current_idx+1}/{len(frame_files)} | {frame_file}"
        # cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show mode info
        # mode_text = f"MODE: {'PLACING POINTS' if editing_mode == 'place_points' else 'VIEWING'}"
        # cv2.putText(display_image, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show class selection
        class_text = f"Mode: {current_class.replace('_', ' ').title()}"
        cv2.putText(display_image, class_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[current_class][0], 2)
        
        # Show polygon count
        polygon_text = f"A: {len(current_polygons)} c: {1 if active_polygon['points'] else 0} a"
        # cv2.putText(display_image, polygon_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show help info
        # cv2.putText(display_image, "Press H for help", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Area Annotation", display_image)
    
    # Create window and set mouse callback
    cv2.namedWindow("Area Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Area Annotation", mouse_callback)
    
    # Function to load a frame and its annotations
    def load_frame(idx):
        nonlocal current_polygons, active_polygon, mouse_x, mouse_y, current_class
        
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
        
        # Reset polygon data when loading a new frame
        current_polygons = []
        
        if existing_annotation:
            # Check for new format with multiple classes
            if "annotations" in existing_annotation:
                for class_annotation in existing_annotation["annotations"]:
                    obj_class = class_annotation["type"]
                    for polygon in class_annotation["polygons"]:
                        current_polygons.append({
                            "class": obj_class,
                            "points": [(int(x), int(y)) for x, y in polygon]
                        })
                print(f"Loaded {len(current_polygons)} areas from multi-class format")
            # Legacy format with just road annotation
            elif "polygons" in existing_annotation:
                for polygon in existing_annotation["polygons"]:
                    current_polygons.append({
                        "class": "drivable_area",
                        "points": [(int(x), int(y)) for x, y in polygon]
                    })
                print(f"Loaded {len(current_polygons)} areas from legacy format")
        else:
            print(f"No existing annotation found for {frame_file}")
        
        active_polygon = {"points": [], "class": current_class}
        
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
            
        # Update the display with the current frame and polygons
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
            # Save current image annotation if polygons exist
            if active_polygon["points"]:
                print("You have an active polygon. Add it (A) or clear it (C) before continuing.")
            elif current_polygons:
                # Add annotation for current frame
                annotation = create_annotation(frame_path, current_polygons)
                
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
                
                if current_idx >= len(frame_files) - 1:
                    print("This is the last image. Annotations saved.")
                else:
                    current_idx += 1
                    current_frame, frame_file, frame_path = load_frame(current_idx)
            else:
                # Check if we're at the last image
                if current_idx >= len(frame_files) - 1:
                    print("This is the last image. No polygons to save.")
                else:
                    current_idx += 1
                    current_frame, frame_file, frame_path = load_frame(current_idx)
                    
        elif key == ord('p'):  # Previous image
            if active_polygon["points"]:
                print("You have an active polygon. Add it (A) or clear it (C) before continuing.")
            else:
                current_idx -= 1
                current_frame, frame_file, frame_path = load_frame(current_idx)
            
        elif key == ord('e'):  # Toggle point placement mode
            editing_mode = "place_points" if editing_mode != "place_points" else None
            print(f"Point placement mode {'activated' if editing_mode == 'place_points' else 'deactivated'}")
            update_display()
            
        elif key == ord('t'):  # Toggle between classes
            class_idx = (available_classes.index(current_class) + 1) % len(available_classes)
            current_class = available_classes[class_idx]
            active_polygon["class"] = current_class  # Update active polygon class
            print(f"Switched to class: {current_class}")
            update_display()
            
        elif key == ord('a'):  # Add current polygon and start a new one
            if active_polygon["points"] and len(active_polygon["points"]) >= 3:  # Need at least 3 points for a polygon
                current_polygons.append(active_polygon.copy())  # Create a copy of the polygon before adding
                print(f"Added {active_polygon['class']} polygon with {len(active_polygon['points'])} points")
                active_polygon = {"points": [], "class": current_class}
                update_display()
            else:
                print("Need at least 3 points to create a polygon")
                
        elif key == ord('f'):  # Finish annotation for current image
            if active_polygon["points"]:
                print("You have an active polygon. Add it (A) or clear it (C) before finishing.")
            else:
                # Create annotation for current frame
                annotation = create_annotation(frame_path, current_polygons)
                
                # Update or add annotation
                found = False
                for i, ann in enumerate(annotations):
                    if ann.get("raw_file") == frame_file:
                        annotations[i] = annotation
                        found = True
                        break
                
                if not found and annotation:
                    annotations.append(annotation)
                
                print(f"Finished annotation for {frame_file}")
                
        elif key == ord('d'):  # Delete last point
            if active_polygon["points"]:
                active_polygon["points"].pop()
                print("Removed last point")
                update_display()
                
        elif key == ord('c'):  # Clear current polygon
            active_polygon["points"] = []
            print("Cleared active polygon")
            update_display()
            
        elif key == ord('r'):  # Reset all polygons for current image
            active_polygon = {"points": [], "class": current_class}
            current_polygons = []
            
            # Remove any existing annotation
            for i, ann in enumerate(annotations):
                if ann.get("raw_file") == frame_file:
                    annotations.pop(i)
                    break
                    
            print("Reset all polygons for current image")
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
            print("T: Toggle between annotation classes (road/car)")
            print("A: Add current polygon and start a new one (needs 3+ points)")
            print("F: Finish annotation for current image and save to JSON")
            print("D: Delete last point placed")
            print("C: Clear current polygon")
            print("R: Reset all polygons on current image")
            print("S: Save annotations to JSON file")
            print("Q: Quit")
            print("\nMOUSE CONTROLS:")
            print("Left click: Place a point (in point mode)")
    
    cv2.destroyAllWindows()
    print("Annotation completed")

if __name__ == "__main__":
    frames_folder = "frames_signals"
    output_json_path = "signals_annotations.json"
    
    create_road_annotations(frames_folder, output_json_path)