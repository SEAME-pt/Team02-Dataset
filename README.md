# Dataset Repository for SEAME Lab

This repository contains datasets and tools for computer vision and autonomous driving research at the SEAME laboratory.

## Overview

This repository hosts two primary datasets:

1. **Local Track Dataset**: Contains frames captured from the physical track along with tools to generate corresponding masks
2. **Carla Simulator Dataset**: Includes frames and masks specifically designed for lane detection algorithms

## Repository Structure

```
Dataset/
├── SEAME/
│   ├── frames/        # Raw image frames from physical track
│   ├── masks/         # Generated mask images for segmentation
│   ├── assets/
│   ├── compare.py   
│   ├── correct.py   
│   ├── frames.py   
│   └── masks.py   
├── carla/
│   ├── frames/             # Frames captured from Carla simulator
│   ├── masks/              # Lane detection masks
│   ├── correct.py/
│   └── CarlaConverter.py/ 
```

## Datasets

### Local Track Dataset

This dataset contains frames captured from our physical test track. The accompanying mask generation program creates segmentation masks that can be used for training computer vision models.

#### Usage

```python
# Example of using the mask generation tool
python masks.py

```

### Carla Simulator Dataset

This dataset is generated from the Carla autonomous driving simulator. It contains both raw frames and corresponding lane masks for lane detection algorithms.

#### Usage

```python
# Example of using the CarlaConverter tool
python CarlaConverter.py --input carla/frames --output carla/masks

```

## Scripts

### SEAME Dataset Scripts
- **frames.py**: Extracts frames from the local track videos
- **masks.py**: Generates masks for the extracted frames
- **compare.py**: Compares generated masks with ground truth
- **correct.py**: Tool for correcting/adjusting masks

### Carla Dataset Scripts
- **CarlaConverter.py**: Converts Carla simulator output to usable dataset format
- **correct.py**: Tool for adjusting and correcting Carla masks

## Future Work

- Add extraction tools for Carla simulator data that can:
  - Extract actor information
  - Generate instance segmentation masks
  - Create labeled datasets for object detection
- Expand datasets with additional driving scenarios
- Create benchmarking tools for evaluation

## Getting Started

1. Clone this repository
2. Install dependencies
3. Run the appropriate scripts to generate masks:
   - For local track: `python SEAME/masks.py`
   - For Carla data: `python carla/CarlaConverter.py`