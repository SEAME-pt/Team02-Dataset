import cv2
import os
import numpy as np

def verify_and_correct_masks(frames_folder, masks_folder):
    """
    Interactive tool to verify and correct lane masks
    
    Controls:
    - N: Next image
    - P: Previous image
    - E: Toggle drawing mode (draw with mouse)
    - D: Toggle erasing mode (erase with mouse)
    - +/-: Increase/decrease brush size (drawing or erasing)
    - C: Clear current mask
    - S: Save changes
    - Q: Quit
    """
    # Get sorted list of all frames
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    # Current frame index
    current_idx = 0
    
    # Mode flags
    editing_mode = None  # None, "draw", or "erase"
    last_point = None
    
    # Tool sizes
    draw_size = 5  # Size of drawing brush (width of line)
    erase_size = 20  # Size of eraser brush
    
    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, current_mask, editing_mode
        
        if editing_mode is None:
            return
            
        # Start drawing or erasing
        if event == cv2.EVENT_LBUTTONDOWN:
            last_point = (x, y)
            
            # Apply immediate effect for eraser
            if editing_mode == "erase":
                cv2.circle(current_mask, (x, y), erase_size, 0, -1)
                # Update display
                display_image = current_frame.copy()
                display_image[current_mask > 0] = (0, 0, 255)  # Red for lane pixels
                # Draw eraser circle
                cv2.circle(display_image, (x, y), erase_size, (0, 255, 255), 2)
                cv2.imshow("Mask Verification", display_image)
            # Apply immediate effect for drawing too
            elif editing_mode == "draw":
                cv2.circle(current_mask, (x, y), draw_size // 2, 255, -1)
                # Update display
                display_image = current_frame.copy()
                display_image[current_mask > 0] = (0, 0, 255)  # Red for lane pixels
                # Draw brush circle
                cv2.circle(display_image, (x, y), draw_size // 2, (255, 255, 0), 2)
                cv2.imshow("Mask Verification", display_image)
            
            changes_made = True
        
        # Continue drawing or erasing when moving with button pressed
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if last_point is not None:
                if editing_mode == "draw":
                    # Draw line with specified thickness
                    cv2.line(current_mask, last_point, (x, y), 255, draw_size)
                    # For thicker lines, draw circles at each point
                    if draw_size > 10:
                        # Calculate intermediate points along the line
                        pts = np.linspace(last_point, (x, y), 10).astype(np.int32)
                        for pt in pts:
                            cv2.circle(current_mask, tuple(pt), draw_size // 2, 255, -1)
                    
                    # Update display with red overlay
                    display_image = current_frame.copy()
                    display_image[current_mask > 0] = (0, 0, 255)
                    # Show brush circle
                    cv2.circle(display_image, (x, y), draw_size // 2, (255, 255, 0), 2)
                    # Show status
                    status_text = f"Frame: {current_idx+1}/{len(frame_files)} | MODE: DRAWING | Size: {draw_size}px"
                    cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif editing_mode == "erase":
                    # Erase with larger radius
                    cv2.circle(current_mask, (x, y), erase_size, 0, -1)
                    # For smoother erasing, draw a line between last point and current
                    if last_point:
                        # Calculate intermediate points along the line
                        pts = np.linspace(last_point, (x, y), 10).astype(np.int32)
                        for pt in pts:
                            cv2.circle(current_mask, tuple(pt), erase_size, 0, -1)
                    
                    # Update display
                    display_image = current_frame.copy()
                    display_image[current_mask > 0] = (0, 0, 255)
                    # Draw eraser circle
                    cv2.circle(display_image, (x, y), erase_size, (0, 255, 255), 2)
                    # Show status
                    status_text = f"Frame: {current_idx+1}/{len(frame_files)} | MODE: ERASING | Size: {erase_size}px"
                    cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                last_point = (x, y)
                cv2.imshow("Mask Verification", display_image)
                changes_made = True
        
        # Release drawing or erasing
        elif event == cv2.EVENT_LBUTTONUP:
            last_point = None
    
    # Create window and set mouse callback
    cv2.namedWindow("Mask Verification", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mask Verification", mouse_callback)
    
    changes_made = False
    
    while True:
        if current_idx < 0:
            current_idx = 0
        if current_idx >= len(frame_files):
            current_idx = len(frame_files) - 1
            
        # Load current frame and mask
        frame_file = frame_files[current_idx]
        frame_path = os.path.join(frames_folder, frame_file)
        mask_path = os.path.join(masks_folder, frame_file.replace('.jpg', '_mask.png'))
        
        current_frame = cv2.imread(frame_path)
        
        # Check if mask exists, if not create an empty one
        if os.path.exists(mask_path):
            current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            current_mask = np.zeros((current_frame.shape[0], current_frame.shape[1]), dtype=np.uint8)
        
        # Create a colored overlay display
        display_image = current_frame.copy()
        display_image[current_mask > 0] = (0, 0, 255)  # Red for lane pixels
        
        # Show status info
        mode_text = "VIEWING"
        brush_info = ""
        if editing_mode == "draw":
            mode_text = "DRAWING"
            brush_info = f" | Size: {draw_size}px"
        elif editing_mode == "erase":
            mode_text = "ERASING"
            brush_info = f" | Size: {erase_size}px"
        
        status_text = f"Frame: {current_idx+1}/{len(frame_files)} | MODE: {mode_text}{brush_info} | Press H for help"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show tool circle preview
        if editing_mode == "erase":
            cv2.putText(display_image, f"Press +/- to adjust eraser size", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif editing_mode == "draw":
            cv2.putText(display_image, f"Press +/- to adjust brush size", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Mask Verification", display_image)
        
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
            
        elif key == ord('p'):  # Previous image
            if changes_made:
                save_confirm = input(f"Save changes to {frame_file}? (y/n): ")
                if save_confirm.lower() == 'y':
                    cv2.imwrite(mask_path, current_mask)
                    print(f"Saved changes to {mask_path}")
                changes_made = False
            current_idx -= 1
            
        elif key == ord('e'):  # Enter drawing mode
            editing_mode = "draw" if editing_mode != "draw" else None
            print(f"Drawing mode {'activated' if editing_mode == 'draw' else 'deactivated'}")
            
        elif key == ord('d'):  # Enter erasing mode
            editing_mode = "erase" if editing_mode != "erase" else None
            print(f"Erasing mode {'activated' if editing_mode == 'erase' else 'deactivated'}")
            
        elif key == ord('+') or key == ord('='):  # Increase tool size
            if editing_mode == "erase":
                erase_size = min(100, erase_size + 5)
                print(f"Eraser size: {erase_size}px")
            elif editing_mode == "draw":
                draw_size = min(50, draw_size + 2)
                print(f"Brush size: {draw_size}px")
            
        elif key == ord('-') or key == ord('_'):  # Decrease tool size
            if editing_mode == "erase":
                erase_size = max(5, erase_size - 5)
                print(f"Eraser size: {erase_size}px")
            elif editing_mode == "draw":
                draw_size = max(1, draw_size - 2)
                print(f"Brush size: {draw_size}px")
            
        elif key == ord('c'):  # Clear mask
            current_mask = np.zeros_like(current_mask)
            display_image = current_frame.copy()
            cv2.imshow("Mask Verification", display_image)
            changes_made = True
            
        elif key == ord('s'):  # Save changes
            cv2.imwrite(mask_path, current_mask)
            print(f"Saved changes to {mask_path}")
            changes_made = False
            
        elif key == ord('h'):  # Help
            print("\nKEYBOARD SHORTCUTS:")
            print("N: Next image")
            print("P: Previous image")
            print("E: Toggle drawing mode (draw with mouse)")
            print("D: Toggle erasing mode (erase with mouse)")
            print("+/-: Increase/decrease brush size (for drawing or erasing)")
            print("C: Clear current mask")
            print("S: Save changes")
            print("Q: Quit")
            print("\nMOUSE CONTROLS:")
            print("Left click + drag: Draw or erase (depending on active mode)")
    
    cv2.destroyAllWindows()
    print("Verification completed")

if __name__ == "__main__":
    frames_folder = "frames"
    masks_folder = "masks"
    
    verify_and_correct_masks(frames_folder, masks_folder)