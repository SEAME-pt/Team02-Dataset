import cv2
import os
import numpy as np
import json
from datetime import datetime

class SignalCutter:
    def __init__(self, frames_folder, output_folder):
        """
        Interactive tool to crop and classify traffic signs from images
        
        Args:
            frames_folder: Path to folder containing source images
            output_folder: Path to SEAMEsignals folder where cropped signs will be saved
        """
        self.frames_folder = frames_folder
        self.output_folder = output_folder
        
        # Class definitions
        self.class_names = [
            "Speed 50km/h",      # 0
            "Speed 80km/h",      # 1  
            "Yield",             # 2
            "Stop",              # 3
            "Danger",            # 4
            "Crosswalk",         # 5
            "Unknown",            # 6
            "Traffic Green",       # 7
            "Traffic Red",       # 7
            "Traffic Yellow"       # 7
        ]
        
        # Create output directories for each class
        self.class_folders = {}
        for i, class_name in enumerate(self.class_names):
            folder_name = f"{i}_{class_name.replace(' ', '_').replace('/', '_')}"
            class_path = os.path.join(output_folder, folder_name)
            os.makedirs(class_path, exist_ok=True)
            self.class_folders[i] = class_path
            print(f"Created/verified folder: {class_path}")
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(frames_folder) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not self.image_files:
            raise ValueError(f"No image files found in {frames_folder}")
        
        print(f"Found {len(self.image_files)} images to process")
        
        # Current state
        self.current_idx = 0
        self.current_image = None
        self.display_image = None
        self.crop_start = None
        self.crop_end = None
        self.is_cropping = False
        self.current_class = 0
        
        # Statistics
        self.crops_saved = {i: 0 for i in range(len(self.class_names))}
        self.total_crops = 0
        
        # Load metadata if exists
        self.metadata_file = os.path.join(output_folder, "crop_metadata.json")
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load existing metadata about cropped images"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"crops": [], "session_info": []}
    
    def save_metadata(self):
        """Save metadata about cropped images"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def load_image(self, idx):
        """Load image at given index"""
        if 0 <= idx < len(self.image_files):
            image_path = os.path.join(self.frames_folder, self.image_files[idx])
            image = cv2.imread(image_path)
            if image is not None:
                self.current_image = image.copy()
                self.display_image = image.copy()
                return True
        return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for cropping selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_start = (x, y)
            self.is_cropping = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.is_cropping:
            # Update display with current selection
            self.display_image = self.current_image.copy()
            cv2.rectangle(self.display_image, self.crop_start, (x, y), (0, 255, 0), 2)
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_cropping:
                self.crop_end = (x, y)
                self.is_cropping = False
                # Draw final rectangle
                self.display_image = self.current_image.copy()
                cv2.rectangle(self.display_image, self.crop_start, self.crop_end, (0, 255, 0), 2)
                self.update_display()
    
    def update_display(self):
        """Update the display with current image and overlays"""
        display = self.display_image.copy()
        
        # Add image info
        image_name = self.image_files[self.current_idx]
        info_text = f"Image: {self.current_idx + 1}/{len(self.image_files)} - {image_name}"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add current class info
        class_text = f"Current Class: {self.current_class} - {self.class_names[self.current_class]}"
        cv2.putText(display, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        
        # Add instructions
        instructions = [
            "SPACE: Save crop with current class",
            "0-6: Change class",
            "N: Next image",
            "P: Previous image",
            "S: Save metadata",
            "H: Show help",
            "Q: Quit"
        ]
        
        start_y = display.shape[0] - (len(instructions) * 20) - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(display, instruction, (10, start_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Signal Cutter", display)
    
    def save_crop(self):
        """Save the currently selected crop with the current class"""
        if self.crop_start is None or self.crop_end is None:
            print("No crop area selected!")
            return False
        
        # Calculate crop coordinates
        x1, y1 = self.crop_start
        x2, y2 = self.crop_end
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Check if crop area is valid
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("Crop area too small!")
            return False
        
        # Extract crop
        crop = self.current_image[y1:y2, x1:x2]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        source_name = os.path.splitext(self.image_files[self.current_idx])[0]
        filename = f"{source_name}_{timestamp}.jpg"
        
        # Save to appropriate class folder
        save_path = os.path.join(self.class_folders[self.current_class], filename)
        
        try:
            cv2.imwrite(save_path, crop)
            
            # Update statistics
            self.crops_saved[self.current_class] += 1
            self.total_crops += 1
            
            # Add to metadata
            crop_info = {
                "filename": filename,
                "source_image": self.image_files[self.current_idx],
                "class_id": self.current_class,
                "class_name": self.class_names[self.current_class],
                "crop_coords": [x1, y1, x2, y2],
                "timestamp": timestamp,
                "crop_size": [x2-x1, y2-y1]
            }
            self.metadata["crops"].append(crop_info)
            
            print(f"Saved crop: {filename} -> Class {self.current_class}: {self.class_names[self.current_class]}")
            
            # Reset crop selection
            self.reset_crop()
            return True
            
        except Exception as e:
            print(f"Error saving crop: {e}")
            return False
    
    def reset_crop(self):
        """Reset crop selection"""
        self.crop_start = None
        self.crop_end = None
        self.display_image = self.current_image.copy()
        self.update_display()
    
    def change_class(self, class_id):
        """Change current class"""
        if 0 <= class_id < len(self.class_names):
            self.current_class = class_id
            print(f"Changed to class {class_id}: {self.class_names[class_id]}")
            self.update_display()
    
    def next_image(self):
        """Load next image"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            if self.load_image(self.current_idx):
                self.reset_crop()
                print(f"Loaded image {self.current_idx + 1}/{len(self.image_files)}")
        else:
            print("Already at last image")
    
    def previous_image(self):
        """Load previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            if self.load_image(self.current_idx):
                self.reset_crop()
                print(f"Loaded image {self.current_idx + 1}/{len(self.image_files)}")
        else:
            print("Already at first image")
    
    def show_help(self):
        """Print help information"""
        print("\n" + "="*60)
        print("SIGNAL CUTTER - HELP")
        print("="*60)
        print("Purpose: Crop traffic signs from images and classify them")
        print()
        print("MOUSE CONTROLS:")
        print("  Left click + drag: Select crop area")
        print()
        print("KEYBOARD CONTROLS:")
        print("  SPACE: Save crop with current class")
        print("  0-6: Change current class")
        print("  N: Next image")
        print("  P: Previous image") 
        print("  R: Reset crop selection")
        print("  S: Save metadata to file")
        print("  H: Show this help")
        print("  Q: Quit application")
        print()
        print("CLASSES:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        print()
        print("CURRENT STATISTICS:")
        for i, count in self.crops_saved.items():
            print(f"  {self.class_names[i]}: {count} crops")
        print(f"  Total: {self.total_crops} crops")
        print("="*60)
    
    def run(self):
        """Main application loop"""
        print("Starting Signal Cutter...")
        print("Use 'H' key for help")
        
        # Load first image
        if not self.load_image(self.current_idx):
            print("Failed to load first image!")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow("Signal Cutter", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Signal Cutter", self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        # Main loop
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
                
            elif key == ord(' '):  # Save crop
                self.save_crop()
                
            elif key == ord('n'):  # Next image
                self.next_image()
                
            elif key == ord('p'):  # Previous image
                self.previous_image()
                
            elif key == ord('r'):  # Reset crop
                self.reset_crop()
                
            elif key == ord('s'):  # Save metadata
                self.save_metadata()
                print("Metadata saved!")
                
            elif key == ord('h'):  # Help
                self.show_help()
                
            elif ord('0') <= key <= ord('9'):  # Change class
                class_id = key - ord('0')
                self.change_class(class_id)
        
        # Save metadata before closing
        self.save_metadata()
        
        # Add session info
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "total_crops": self.total_crops,
            "crops_per_class": dict(self.crops_saved),
            "images_processed": self.current_idx + 1
        }
        self.metadata["session_info"].append(session_info)
        self.save_metadata()
        
        cv2.destroyAllWindows()
        
        print(f"\nSession completed!")
        print(f"Total crops saved: {self.total_crops}")
        for i, count in self.crops_saved.items():
            if count > 0:
                print(f"  {self.class_names[i]}: {count}")

def main():
    """Main function to run the signal cutter"""
    # Set up paths
    frames_folder = "framesTrafficLights"
    signals_folder = "SEAMEsignals"
    
    # Check if frames folder exists
    if not os.path.exists(frames_folder):
        print(f"Error: Frames folder '{frames_folder}' not found!")
        print("Please make sure you're running this from the SEAME directory")
        return
    
    # Create signals folder if it doesn't exist
    os.makedirs(signals_folder, exist_ok=True)
    
    try:
        # Create and run the signal cutter
        cutter = SignalCutter(frames_folder, signals_folder)
        cutter.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
