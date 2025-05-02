# Dataset Repository for SEAME Lab

This repository contains datasets and tools for computer vision and autonomous driving research at the SEAME laboratory.

## Overview

This repository hosts two primary datasets:

1. **Local Track Dataset**: Contains frames captured from the physical track along with tools to generate corresponding masks
2. **Carla Simulator Dataset**: Includes frames and masks specifically designed for lane detection algorithms

## Repository Structure

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