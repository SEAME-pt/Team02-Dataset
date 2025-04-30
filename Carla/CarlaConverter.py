#!/usr/bin/env python3
# filepath: /Users/ruipedropires/SEAME/carla_reorganize_dataset.py

import os
import shutil
from pathlib import Path
import re
import argparse
from tqdm import tqdm
from PIL import Image

def reorganize_carla_dataset(source_dir, target_dir):
    """
    Reorganize Carla dataset files from:
        - Carla/rgb/00003600.png -> frames/frame_00003600.jpg
        - Carla/lane/lane_003600.png -> masks/frame_00003600_mask.png
    
    Args:
        source_dir (str): Path to the source directory containing the Carla folder
        target_dir (str): Path to the target directory where reorganized files will be saved
    """
    # Create target directories if they don't exist
    frames_dir = os.path.join(target_dir, 'frames')
    masks_dir = os.path.join(target_dir, 'masks')
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"Created output directories: {frames_dir} and {masks_dir}")
    
    # Process images in 'Carla/rgb' directory
    rgb_dir = os.path.join(source_dir, 'Carla', 'rgb')
    if os.path.exists(rgb_dir):
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        print(f"Found {len(rgb_files)} image files in 'Carla/rgb' directory")
        
        for file in tqdm(rgb_files, desc="Processing RGB images"):
            # Extract number from filename (e.g., '00003600' from '00003600.png')
            match = re.search(r'(\d+)\.png', file)
            if match:
                num = match.group(1)
                
                # Create new filename
                new_filename = f"frame_{num}.jpg"
                
                # Convert PNG to JPG using PIL
                src_path = os.path.join(rgb_dir, file)
                dst_path = os.path.join(frames_dir, new_filename)
                
                try:
                    # Open image, convert to RGB, and save as JPG
                    img = Image.open(src_path)
                    img = img.convert('RGB')
                    img.save(dst_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error converting {src_path}: {e}")
                    # Fallback to simple copy if conversion fails
                    shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: Source directory 'Carla/rgb' not found at {rgb_dir}")
    
    # Process masks in 'Carla/lane' directory
    lane_dir = os.path.join(source_dir, 'Carla', 'lanes')
    if os.path.exists(lane_dir):
        mask_files = [f for f in os.listdir(lane_dir) if f.startswith('lane_') and f.endswith('.png')]
        print(f"Found {len(mask_files)} mask files in 'Carla/lane' directory")
        
        for file in tqdm(mask_files, desc="Processing lane masks"):
            # Extract number from filename (e.g., '003600' from 'lane_003600.png')
            match = re.search(r'lane_(\d+)\.png', file)
            if match:
                num = match.group(1)
                
                # Pad to ensure consistent formatting
                padded_num = num.zfill(8)
                
                # Create new filename
                new_filename = f"frame_{padded_num}_mask.png"
                
                # Copy file
                src_path = os.path.join(lane_dir, file)
                dst_path = os.path.join(masks_dir, new_filename)
                shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: Source directory 'Carla/lane' not found at {lane_dir}")
    
    # Check for mismatches between frames and masks
    frame_numbers = set()
    mask_numbers = set()
    
    for file in os.listdir(frames_dir):
        if file.startswith('frame_') and file.endswith('.jpg'):
            match = re.search(r'frame_(\d+)\.jpg', file)
            if match:
                frame_numbers.add(match.group(1))
    
    for file in os.listdir(masks_dir):
        if file.endswith('_mask.png'):
            match = re.search(r'frame_(\d+)_mask\.png', file)
            if match:
                mask_numbers.add(match.group(1))
    
    frames_only = frame_numbers - mask_numbers
    masks_only = mask_numbers - frame_numbers
    
    if frames_only:
        print(f"Warning: {len(frames_only)} frames have no corresponding mask")
    
    if masks_only:
        print(f"Warning: {len(masks_only)} masks have no corresponding frame")
    
    print("Reorganization complete!")
    print(f"Images copied to: {frames_dir}")
    print(f"Masks copied to: {masks_dir}")
    
    # Verify the counts
    final_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    final_masks = len([f for f in os.listdir(masks_dir) if f.endswith('_mask.png')])
    print(f"Final count: {final_frames} images, {final_masks} masks")

def main():
    parser = argparse.ArgumentParser(description='Reorganize Carla dataset files.')
    parser.add_argument('source_dir', type=str, help='Path to the source directory containing the Carla folder')
    parser.add_argument('target_dir', type=str, help='Path to the target directory where reorganized files will be saved')
    args = parser.parse_args()
    
    # Run the reorganization
    reorganize_carla_dataset(args.source_dir, args.target_dir)

if __name__ == '__main__':
    main()