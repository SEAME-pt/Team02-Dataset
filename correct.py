import cv2
import os
import numpy as np

def verify_and_correct_masks(frames_folder, masks_folder):
    """
    Interactive tool to verify and correct lane masks
    
    Controls:
    - N: Next image
    - P: Previous image
    - E: Toggle point placement mode
    - L: Connect placed points with a line
    - D: Toggle erasing mode (erase with mouse)
    - +/-: Increase/decrease point/eraser size
    - C: Clear current mask or points
    - R: Reset placed points (keep mask)
    - S: Save changes
    - Q: Quit
    """
    # Get sorted list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    # Filter to only include frames that have corresponding masks
    frames_with_masks = []
    for frame_file in frame_files:
        mask_path = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        if os.path.exists(mask_path):
            frames_with_masks.append(frame_file)
    
    if not frames_with_masks:
        print("No frames with corresponding masks found!")
        return
    
    print(f"Found {len(frames_with_masks)} frames with masks out of {len(frame_files)} total frames")
    
    # Current frame index
    current_idx = 0
    
    # Mode flags
    editing_mode = None  # None, "place_points", or "erase"
    
    # Points collection for line drawing - store (x, y, size)
    placed_points = []
    
    # Tool sizes
    current_point_size = 8  # Size of next point to place
    erase_size = 20  # Size of eraser brush
    
    # Tracking variables
    changes_made = False
    current_frame = None
    current_mask = None
    
    # Mouse position tracking for preview
    mouse_x, mouse_y = -1, -1
    
    # Function to draw line between points
    def connect_points_with_line():
        nonlocal current_mask, changes_made
        
        if len(placed_points) < 2:
            print("Need at least 2 points to create a line")
            return False
        
        print(f"Starting line drawing with {len(placed_points)} points")
        
        # Make a backup copy of the current mask for verification
        original_mask = current_mask.copy()
        original_white_pixels = np.sum(original_mask > 0)
        
        # Draw directly on the current mask
        for i in range(len(placed_points) - 1):
            x1, y1, size1 = placed_points[i]
            x2, y2, size2 = placed_points[i + 1]
            
            # Calculate line thickness to exactly match point sizes
            # Use the full size, not half size, to ensure consistency
            start_thickness = size1  # Full diameter of first point
            end_thickness = size2    # Full diameter of second point
            
            # For consistent thickness lines, use the max of the two sizes
            line_thickness = max(start_thickness, end_thickness)
            
            # Draw the main line with the calculated thickness
            cv2.line(current_mask, (x1, y1), (x2, y2), 255, thickness=line_thickness)
            
            # Draw endpoints as filled circles with full size diameter
            cv2.circle(current_mask, (x1, y1), size1 // 2, 255, -1)
            cv2.circle(current_mask, (x2, y2), size2 // 2, 255, -1)
            
            # If the two points have very different sizes, add a smooth transition
            if abs(size1 - size2) > 4:
                # Calculate distance between points
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                num_steps = max(10, int(distance / 5))
                
                # Create a gradual transition along the line
                for t in np.linspace(0, 1, num_steps):
                    # Calculate position along the line
                    x = int((1-t) * x1 + t * x2)
                    y = int((1-t) * y1 + t * y2)
                    
                    # Calculate interpolated size
                    interp_size = int((1-t) * size1 + t * size2)
                    
                    # Draw a circle at this position with interpolated radius
                    cv2.circle(current_mask, (x, y), interp_size // 2, 255, -1)
        
        # Count white pixels to verify drawing worked
        new_white_pixels = np.sum(current_mask > 0)
        print(f"Mask had {original_white_pixels} white pixels, now has {new_white_pixels}")
        
        if new_white_pixels <= original_white_pixels:
            print("WARNING: Drawing didn't add any new pixels. Something might be wrong.")
        
        # Apply a small amount of smoothing to make sure everything blends well
        kernel = np.ones((5, 5), np.uint8)
        current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
        
        # Immediately save the mask to ensure changes are preserved
        frame_file = frames_with_masks[current_idx]
        mask_path = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        success = cv2.imwrite(mask_path, current_mask)
        
        if success:
            print(f"Automatically saved mask with line to {mask_path}")
        else:
            print(f"ERROR: Failed to save mask to {mask_path}")
        
        # Force update the display to show changes
        update_display()
        
        print(f"Connected {len(placed_points)} points with matching thickness")
        return True
    
    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal placed_points, current_mask, editing_mode, changes_made, current_point_size
        nonlocal mouse_x, mouse_y
        
        # Update mouse position for preview
        mouse_x, mouse_y = x, y
        
        if editing_mode is None:
            return
        
        # Update display to show preview at current position
        if editing_mode == "place_points" or editing_mode == "erase":
            update_display()
            
        # Place point or start erasing
        if event == cv2.EVENT_LBUTTONDOWN:
            if editing_mode == "place_points":
                # Add point to the collection with its size
                placed_points.append((x, y, current_point_size))
                print(f"Placed point #{len(placed_points)} at ({x}, {y}) with size {current_point_size}")
                update_display()
                
            elif editing_mode == "erase":
                # Erase in a circle
                cv2.circle(current_mask, (x, y), erase_size, 0, -1)
                changes_made = True
                update_display()
                
        # Continue erasing when moving with button pressed
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if editing_mode == "erase":
                # Erase with larger radius
                cv2.circle(current_mask, (x, y), erase_size, 0, -1)
                changes_made = True
                update_display()
    
    # Function to update the display
    def update_display():
        nonlocal current_frame, current_mask, placed_points, editing_mode, current_point_size
        nonlocal mouse_x, mouse_y
        
        # Create a colored overlay display
        display_image = current_frame.copy()
        
        # Apply the mask with red overlay
        mask_pixels = current_mask > 0
        if np.any(mask_pixels):
            display_image[mask_pixels] = (0, 0, 255)  # BGR format - Red
        
        # Show original filename for reference
        frame_file = frames_with_masks[current_idx]
        original_frame_number = frame_file.split('_')[1].split('.')[0]
        
        # Show status info
        mode_text = "VIEWING"
        tool_info = ""
        if editing_mode == "place_points":
            mode_text = "PLACING POINTS"
            tool_info = f" | Current Size: {current_point_size}px | Points: {len(placed_points)}"
        elif editing_mode == "erase":
            mode_text = "ERASING"
            tool_info = f" | Size: {erase_size}px"
        
        status_text = f"Frame: {current_idx+1}/{len(frames_with_masks)} | Frame #{original_frame_number} | MODE: {mode_text}{tool_info}"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show help info
        cv2.putText(display_image, "Press H for help", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show mode-specific instructions
        if editing_mode == "place_points":
            cv2.putText(display_image, "Click to place points, press L to connect with line", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_image, f"+/- to adjust size BEFORE placing next point, R to reset", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw all placed points
            for i, (x, y, size) in enumerate(placed_points):
                # Draw the point
                cv2.circle(display_image, (x, y), size // 2, (0, 255, 255), -1)
                # Draw the point number and size
                cv2.putText(display_image, f"{i+1}:{size}", (x+5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw preview point at current mouse position if mouse is in the window
            if mouse_x >= 0 and mouse_y >= 0 and mouse_x < display_image.shape[1] and mouse_y < display_image.shape[0]:
                # Draw transparent preview circle
                preview_img = display_image.copy()
                
                # Draw dashed circle outline to make it visible on any background
                # Draw a dotted circle by drawing short arcs
                num_segments = 16
                for i in range(0, num_segments, 2):
                    angle1 = i * 360 / num_segments
                    angle2 = (i + 1) * 360 / num_segments
                    cv2.ellipse(display_image, (mouse_x, mouse_y), 
                               (current_point_size // 2, current_point_size // 2),
                               0, angle1, angle2, (0, 255, 255), 2)
                
                # Draw semi-transparent fill
                cv2.circle(preview_img, (mouse_x, mouse_y), current_point_size // 2, (0, 255, 255), -1)
                # Blend with original image
                alpha = 0.4  # Transparency factor
                cv2.addWeighted(preview_img, alpha, display_image, 1 - alpha, 0, display_image)
                
                # Show size info near cursor
                cv2.putText(display_image, f"Size: {current_point_size}px", 
                           (mouse_x + 15, mouse_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        elif editing_mode == "erase":
            cv2.putText(display_image, f"Drag to erase, +/- to adjust eraser size", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show eraser preview at mouse position
            if mouse_x >= 0 and mouse_y >= 0:
                cv2.circle(display_image, (mouse_x, mouse_y), erase_size, (0, 255, 255), 2)
        
        cv2.imshow("Mask Verification", display_image)
    
    # Create window and set mouse callback
    cv2.namedWindow("Mask Verification", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mask Verification", mouse_callback)
    
    while True:
        if current_idx < 0:
            current_idx = 0
        if current_idx >= len(frames_with_masks):
            current_idx = len(frames_with_masks) - 1
            
        # Load current frame and mask
        frame_file = frames_with_masks[current_idx]
        frame_path = os.path.join(frames_folder, frame_file)
        mask_path = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        
        current_frame = cv2.imread(frame_path)
        
        # Load the mask (we already confirmed it exists)
        current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Reset mouse position when loading a new image
        mouse_x, mouse_y = -1, -1
        
        # Update the display with the current frame and mask
        update_display()
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit
            if changes_made:
                save_confirm = input("You have unsaved changes. Save before quitting? (y/n): ")
                if save_confirm.lower() == 'y':
                    cv2.imwrite(mask_path, current_mask)
                    print(f"Saved changes to {mask_path}")
            break
            
        elif key == ord('n'):  # Next image
            if changes_made:
                save_confirm = input(f"Save changes to {frame_file}? (y/n): ")
                if save_confirm.lower() == 'y':
                    cv2.imwrite(mask_path, current_mask)
                    print(f"Saved changes to {mask_path}")
                changes_made = False
            current_idx += 1
            placed_points = []  # Reset points when moving to next image
            
        elif key == ord('p'):  # Previous image
            if changes_made:
                save_confirm = input(f"Save changes to {frame_file}? (y/n): ")
                if save_confirm.lower() == 'y':
                    cv2.imwrite(mask_path, current_mask)
                    print(f"Saved changes to {mask_path}")
                changes_made = False
            current_idx -= 1
            placed_points = []  # Reset points when moving to previous image
            
        elif key == ord('e'):  # Enter point placement mode
            editing_mode = "place_points" if editing_mode != "place_points" else None
            print(f"Point placement mode {'activated' if editing_mode == 'place_points' else 'deactivated'}")
            update_display()
            
        elif key == ord('l'):  # Connect points with a line
            if editing_mode == "place_points":
                if connect_points_with_line():
                    changes_made = True
                    update_display()
            
        elif key == ord('d'):  # Enter erasing mode
            editing_mode = "erase" if editing_mode != "erase" else None
            print(f"Erasing mode {'activated' if editing_mode == 'erase' else 'deactivated'}")
            update_display()
            
        elif key == ord('r'):  # Reset points (but keep mask)
            placed_points = []
            print("Reset all placed points")
            update_display()
            
        elif key == ord('+') or key == ord('='):  # Increase tool size
            if editing_mode == "erase":
                erase_size = min(100, erase_size + 5)
                print(f"Eraser size: {erase_size}px")
            elif editing_mode == "place_points":
                current_point_size = min(30, current_point_size + 2)
                print(f"Next point size: {current_point_size}px")
            update_display()
            
        elif key == ord('-') or key == ord('_'):  # Decrease tool size
            if editing_mode == "erase":
                erase_size = max(5, erase_size - 5)
                print(f"Eraser size: {erase_size}px")
            elif editing_mode == "place_points":
                current_point_size = max(2, current_point_size - 2)
                print(f"Next point size: {current_point_size}px")
            update_display()
            
        elif key == ord('c'):  # Clear mask
            if editing_mode == "place_points":
                placed_points = []
                print("Reset all placed points")
            else:
                current_mask = np.zeros_like(current_mask)
                changes_made = True
                print("Cleared mask")
            update_display()
            
        elif key == ord('s'):  # Save changes
            cv2.imwrite(mask_path, current_mask)
            print(f"Saved changes to {mask_path}")
            changes_made = False
            
        elif key == ord('h'):  # Help
            print("\nKEYBOARD SHORTCUTS:")
            print("N: Next image")
            print("P: Previous image")
            print("E: Toggle point placement mode")
            print("L: Connect placed points with a line")
            print("D: Toggle erasing mode (erase with mouse)")
            print("R: Reset placed points (keep mask)")
            print("+/-: Adjust size for next point or eraser")
            print("C: Clear mask or points (depends on mode)")
            print("S: Save changes")
            print("Q: Quit")
            print("\nMOUSE CONTROLS:")
            print("Left click: Place a point (in point mode) or erase (in erase mode)")
    
    cv2.destroyAllWindows()
    print("Verification completed")

if __name__ == "__main__":
    frames_folder = "frames"
    masks_folder = "masks"
    
    verify_and_correct_masks(frames_folder, masks_folder)