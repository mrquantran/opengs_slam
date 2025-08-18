#!/bin/bash

# Installation script for OpenGS-SLAM with PGO (Pose Graph Optimization)
# This script installs the required dependencies for the PGO module

echo "Installing OpenGS-SLAM with PGO dependencies..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating/updating environment..."

    # Create or update conda environment
    if conda env list | grep -q "opengs_slam"; then
        echo "Updating existing opengs_slam environment..."
        conda env update -f environment.yml
    else
        echo "Creating new opengs_slam environment..."
        conda env create -f environment.yml
    fi

    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate opengs_slam
else
    echo "Conda not found. Please install conda first or use pip directly."
    exit 1
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements_pgo.txt

# Install YOLOv8
echo "Installing YOLOv8..."
pip install ultralytics

# Try to install g2o-python (this might fail and need manual installation)
echo "Attempting to install g2o-python..."
pip install g2o-python || {
    echo "Warning: g2o-python installation failed."
    echo "You may need to install it manually from source:"
    echo "1. Clone https://github.com/uoip/g2opy"
    echo "2. Follow the installation instructions in the repository"
    echo "3. Or use an alternative like GTSAM or Ceres"
}

# Download YOLOv8 model weights
echo "Downloading YOLOv8 model weights..."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8l-seg.pt')
    print('YOLOv8 model downloaded successfully')
except Exception as e:
    print(f'Error downloading YOLOv8 model: {e}')
    print('You may need to download it manually')
"

echo ""
echo "Installation completed!"
echo ""
echo "Next steps:"
echo "1. If g2o-python installation failed, install it manually"
echo "2. Test the installation by running: python -c 'from utils.slam_pgo import PGOThread; print(\"PGO module imported successfully\")'"
echo "3. Run your SLAM system with PGO enabled"
echo ""
echo "Note: The PGO module will work even without g2o-python, but pose graph optimization will be disabled."
