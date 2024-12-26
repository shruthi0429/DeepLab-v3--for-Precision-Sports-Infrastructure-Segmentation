# DeepLab-v3-for-Precision-Sports-Infrastructure-Segmentationn



## Introduction
This project implements an advanced semantic image segmentation system utilizing the DeepLab v3+ architecture. The primary objective is to accurately detect and segment tennis courts and swimming pools from both aerial and ground-level imagery. The implementation leverages transfer learning techniques and custom optimization strategies to achieve high segmentation accuracy.

## Dataset 
### Overview
We created a custom dataset specifically designed for sports facility segmentation:
- **Generation Method**: Utilized Stable Diffusion Model to generate diverse and representative images
- **Annotation Tool**: Labelme for precise ground truth segmentation masks
- **Total Dataset Size**: 200 high-quality annotated images
- **Class Distribution**: 3 distinct classes for comprehensive facility segmentation

### Data Split Strategy
Implemented a standard ML split methodology:
- **Training Set**: 160 images (80%) - Used for model learning
- **Validation Set**: 20 images (10%) - Used for hyperparameter tuning
- **Test Set**: 20 images (10%) - Used for final model evaluation

## Data Processing Pipeline
### Normalization
Implemented standardized normalization following ImageNet statistics:
- **Mean Values**: [0.485, 0.456, 0.406]
- **Standard Deviation**: [0.229, 0.224, 0.225]

### Data Augmentation Suite
Comprehensive augmentation strategy to enhance model robustness:
- **Spatial Transformations**:
  - Random scaling for size invariance
  - Strategic cropping for focus variation
  - Horizontal and vertical flipping for orientation diversity
  - Random rotation for perspective invariance

## Model Architecture
### Base Architecture
- **Framework**: DeepLab v3+
- **Key Features**:
  - Atrous Spatial Pyramid Pooling (ASPP)
  - Encoder-Decoder structure
  - Deep residual learning
  - Skip connections for fine detail preservation

### Loss Function
- Implemented Cross Entropy Loss for multi-class segmentation
- Optimized for class imbalance handling

## Training Framework
### Experimental Configurations
Conducted three distinct experiments with varying batch sizes:
- **Experiment 1**: Batch size 8
- **Experiment 2**: Batch size 16
- **Experiment 3**: Batch size 4

### Optimization Strategy
- **Algorithm**: Stochastic Gradient Descent (SGD)
- **Fine-tuning Parameters**:
  - Weight Decay: 0.0001 for regularization
  - Output Stride: Dynamic adjustment [16, 8, 16]
  - Fixed Crop Size: 513 pixels
  - Random Seed: 1 for reproducibility

## Data Structure Specifications
### Segmentation Format
- **Base Format**: VOC-style segmentation masks
- **Encoding Scheme**:
  - Color Encoding for Swimming Pool: Blue [0, 0, 255]
  - Numerical Class Encoding for Swimming Pool: 1

## Transfer Learning Implementation
### Strategy
```python
# Backbone Layer Freezing
for param in model.backbone.parameters():
    param.requires_grad = False

# Additional optimization configurations
if opts.separable_conv and 'plus' in opts.model:
    network.convert_to_separable_conv(model.classifier)
utils.set_bn_momentum(model.backbone, momentum=0.01)
```

## Conclusions and Results
The implementation demonstrated significant improvements in segmentation accuracy through:
- Effective transfer learning utilization
- Optimized hyperparameter selection
- Robust data augmentation strategy
- Custom-tuned architecture modifications

## Future Work
Potential areas for improvement:
- Integration of additional sports facility types
- Enhancement of boundary detection accuracy
- Implementation of real-time segmentation capabilities
- Exploration of alternative backbone architectures

## Dependencies
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Labelme (for annotation)
![image](https://github.com/user-attachments/assets/4454f085-0676-41d8-b25d-513f27803148)

![image](https://github.com/user-attachments/assets/74382a74-7f4c-49f7-8b5c-7269cc8e1721)
