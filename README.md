# SEAME and Carla Dataset Tools

This repository contains tools for creating and using autonomous driving datasets from two sources:
1. SEAME - A custom dataset for lane and road segmentation
2. Carla - Synthetic data from the Carla simulator

## SEAME Dataset Tools

This section contains tools for creating and using the SEAME dataset for autonomous driving tasks, including lane detection and road segmentation.

### Annotation Tools

#### Lane Annotation Tool (`lane_annotations.py`)

An interactive tool for creating lane annotations in TuSimple format.

**Features:**
- Mark individual lane points with precise placement
- Visual feedback with lane previews
- Automatic export to TuSimple-compatible format
- Ability to save and resume annotation work

**Usage:**
```bash
python lane_annotations.py
```

**Controls:**
- N: Next image
- P: Previous image
- E: Toggle point placement mode
- A: Add current lane and start a new one
- F: Finish annotation for current image
- D: Delete last point placed
- C: Clear current lane
- R: Reset all lanes on current image
- S: Save annotations to JSON file
- Q: Quit

#### Road and Object Annotation Tool (`road_annotations.py`)

An interactive tool for creating polygon annotations for drivable areas and objects (cars).

**Features:**
- Create polygon annotations for multiple classes
- Toggle between classes (drivable_area, car)
- Visual feedback with semi-transparent overlays
- Saves annotations in JSONL format

**Usage:**
```bash
python road_annotations.py
```

**Controls:**
- N: Next image
- P: Previous image
- E: Toggle polygon point placement mode
- T: Toggle between annotation classes (road/car)
- A: Add current polygon and start a new one
- F: Finish annotation for current image
- D: Delete last point placed
- C: Clear current polygon
- R: Reset all polygons on current image
- S: Save annotations to JSON file
- Q: Quit

### Dataset Implementations

#### Lane Detection Dataset (`SEAMELaneDataset.py`)

A PyTorch Dataset implementation for lane detection tasks.

**Features:**
- Loads TuSimple-format lane annotations
- Provides binary lane segmentation labels
- Applies data augmentation for training
- Compatible with PyTorch DataLoader

**Example Usage:**
```python
from torch.utils.data import DataLoader
from SEAMELaneDataset import SEAMEDataset

# Initialize dataset
train_dataset = SEAMEDataset(
    json_paths=['lane_annotations.json'],
    img_dir='frames',
    width=512,
    height=256,
    is_train=True
)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Use in training loop
for images, binary_labels in train_loader:
    # Your model training code
    pass
```

#### Road Segmentation Dataset (`SEAMERoadDataset.py`)

A PyTorch Dataset implementation for multi-class segmentation tasks.

**Features:**
- Supports multiple classes (background, drivable_area, car)
- Loads polygon annotations
- Applies data augmentation for training
- Includes visualization tools
- Compatible with PyTorch DataLoader

**Example Usage:**
```python
from torch.utils.data import DataLoader
from SEAMERoadDataset import SEAMESegmentationDataset

# Initialize dataset
train_dataset = SEAMESegmentationDataset(
    img_dir='frames',
    annotation_file='road_annotations.json',
    width=256,
    height=128,
    is_train=True
)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Use in training loop
for images, masks in train_loader:
    # Your model training code
    pass

# Visualize annotations
dataset.visualize(0)  # Visualize the first sample
```

## Carla Dataset Tools

This section contains tools for creating synthetic datasets using the Carla simulator.

### Annotation Tools

#### Lane Annotation Tool (`lane_annotations.py`)

Similar to SEAME's lane annotation tool, this is an interactive tool for creating lane annotations from Carla simulation frames.

**Features:**
- Mark individual lane points with precise placement
- Visual feedback with lane previews
- Automatic export to TuSimple-compatible format
- Enhanced visualization modes
- Ability to save and resume annotation work

**Usage:**
```bash
python lane_annotations.py
```

The tool expects frames in the `lane_dataset/frames` directory and will save annotations to `lane_dataset/lane_annotations.json`.

**Controls:**
- N: Next image
- P: Previous image
- E: Toggle point placement mode
- A: Add current lane and start a new one
- F: Finish annotation for current image
- D: Delete last point placed
- C: Clear current lane
- R: Reset all lanes on current image
- S: Save annotations to JSON file
- H: Show help
- Q: Quit

#### Object Annotation Tool (`object_annotations.py`)

Automated tool that connects to the Carla simulator to capture RGB images and semantic segmentation masks for object detection and segmentation tasks.

**Features:**
- Direct integration with Carla simulator
- Automatic capture of RGB images with corresponding semantic segmentation masks
- Configurable capture frequency
- Autopilot driving for varied scene capture
- Semantic tag extraction for multiple object classes
- No manual annotation required - uses Carla's built-in segmentation

**Usage:**
```bash
python object_annotations.py
```

The tool will:
1. Connect to the Carla simulator (must be running)
2. Spawn a vehicle with attached RGB and semantic segmentation cameras
3. Enable autopilot for automated driving
4. Capture frames at specified intervals
5. Save RGB images to `obj_dataset/images/`
6. Save semantic masks to `obj_dataset/masks/`

**Requirements:**
- Carla simulator (tested with version 0.9.13)
- Pygame for visualization
- Properly configured Carla Python API

## Dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

For Carla tools, you'll also need to set up the Carla simulator and Python API as described in the [Carla documentation](https://carla.readthedocs.io/).