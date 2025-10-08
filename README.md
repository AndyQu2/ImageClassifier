# Image Classification with AlexNet

## Overview
This project implements an image classification system using the AlexNet deep learning architecture. It includes both 
model training capabilities and a user-friendly GUI application for image classification tasks. The system is designed 
to handle custom datasets and provides tools for training, evaluation, and inference.

## Features
- AlexNet implementation with PyTorch
- Custom dataset loader with support for common image formats
- Training pipeline with automatic mixed precision (AMP) and gradient clipping
- GUI interface built with wxPython for easy image classification
- Model checkpointing and best model selection

## Project Structure
    ├── datasets            # Sample images which are used to train the model
    ├── alexnet.py          # AlexNet model definition and training utilities
    ├── data_loader.py      # Custom Dataset class for image loading
    ├── main_frame.py       # GUI frame implementation
    ├── main_window.py      # Main window class inheriting from main_frame
    ├── main.py             # Application entry point
    └── train.py            # Model training script 

## Requirements
- Python 3.7+
- PyTorch (with CUDA support if available)
- torchvision
- wxPython
- PIL (Pillow)
- numpy

## Installation
1. Clone this repository
2. Install required dependencies: ``pip install torch torchvision wxPython Pillow numpy``

## Usage

### Training the Model
1. Organize your dataset in the following structure:
```
datasets/
   class1/
     image1.jpg
     image2.jpg
   ...
   class2/
     image1.jpg
```
2. Run the training script: ```python main.py```
3. The script will automatically:
- Load and preprocess images
- Initialize AlexNet with Xavier initialization
- Train with SGD optimizer (momentum=0.9, weight_decay=0.0005)
- Use learning rate scheduling (ExponentialLR with gamma=0.95)
- Save the best model to `model.pth`

### Using the GUI Application
1. Launch the application
2. 2. Use the interface to:
- Load images using the "Load Image" button
- Classify images with the "Classify Image" button
- View classification results and confidence scores

## Model Architecture
The implementation uses a modified AlexNet with the following structure:
- Convolutional layers with ReLU activation and max pooling
- Adaptive average pooling to fixed size (6×6)
- Fully connected layers with dropout regularization
- Final output layer with 79 classes (customizable)

## Training Details
- Batch size: 256
- Learning rate: 0.1 (with exponential decay)
- Epochs: 100
- Loss function: CrossEntropyLoss
- Optimization: SGD with momentum
- Gradient clipping: max_norm=1.0

## Performance Features
- Automatic mixed precision training when CUDA is available
- Gradient scaling and clipping for stable training
- Best model checkpointing based on validation accuracy
- Multithreaded data loading with prefetching

## Notes
- The model expects input images of size 224×224 pixels
- Images are automatically resized during both training and inference
- The number of output classes (79) can be modified in the AlexNet definition
- For best performance, use a GPU-enabled environment

## License
This project is provided for educational and research purposes. Please ensure you have appropriate rights for any datasets used with this software.
