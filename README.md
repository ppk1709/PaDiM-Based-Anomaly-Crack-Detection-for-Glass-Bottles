# PaDiM-Based-Anomaly-Crack-Detection-for-Glass-Bottles


# Bottle Inspector - Crack Detection System

A comprehensive automated visual inspection system that detects cracks and defects in bottles using deep learning and computer vision. This system combines PaDiM (Patch Distribution Modeling) anomaly detection with industrial automation controls for real-time quality inspection.

![System Overview](https://img.shields.io/badge/Status-Production_Ready-green.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-PaDiM-orange.svg)

## 🎯 How It Detects Cracks in Bottles

### Technology Overview

This system uses **PaDiM (Patch Distribution Modeling)**, an unsupervised anomaly detection approach that:

1. **Learns Normal Appearance**: The system is trained on "good" bottles without cracks to learn the normal visual patterns
2. **Detects Anomalies**: During inspection, it identifies deviations from the learned normal patterns
3. **Patch-based Analysis**: Breaks the image into small patches and models their feature distributions

### Crack Detection Process

#### Training Phase:
- **Collect Normal Samples**: Capture multiple images of defect-free bottles
- **Feature Extraction**: Uses ResNet-18 to extract deep features from image patches
- **Statistical Modeling**: Builds Gaussian distributions of normal feature patterns
- **Model Saving**: Saves the statistical model for each bottle variant

#### Inspection Phase:
- **Real-time Capture**: Captures bottle images on the production line
- **Feature Comparison**: Compares current bottle features against trained normal distributions
- **Anomaly Scoring**: Computes Mahalanobis distance to identify statistical outliers
- **Decision Making**: Flags bottles with anomaly scores above threshold as defective

### What Makes It Effective for Crack Detection

1. **High Sensitivity**: Detects subtle texture changes caused by micro-cracks
2. **Unsupervised Learning**: Doesn't require defective samples for training
3. **Multi-scale Features**: Analyzes both local details and global patterns
4. **Robust to Variations**: Handles normal appearance variations in bottles

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- USB Camera
- Modbus-compatible PLC (optional for automation)
- Windows/Linux/macOS

### Installation

1. **Clone or Download the Project Files**
   ```bash
   # Create project directory
   mkdir bottle_inspector
   cd bottle_inspector
   ```

2. **Save the Provided Files**
   - `resnet_feature_extractor.py`
   - `inspector_app.py` 
   - `main.py`
   - `requirements.txt`

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python main.py
   ```

### Hardware Setup

```
[Production Line] → [Sensor] → [Camera] → [Computer] → [PLC] → [Accept/Reject Mechanism]
      │              │          │           │           │
      Bottle      Detection   Image      Processing  Control
```

## 📋 Step-by-Step Running Instructions

### Step 1: Initial Setup

1. **Connect Camera**: Plug in your USB camera to the computer
2. **Check Camera Index**: Modify `CAM_INDEX` in `inspector_app.py` if needed (default: 0)
3. **PLC Connection** (Optional): Set correct COM port in `SERIAL_PORT` variable

### Step 2: Configure Inspection Parameters

1. **Launch Application**:
   ```bash
   python main.py
   ```

2. **Set ROI (Region of Interest)**:
   - Click "Set ROI" button
   - Drag on the camera preview to select bottle area
   - Release to set the inspection region

3. **Configure Enhancement** (Optional):
   - Adjust Alpha, Beta, Gamma for contrast
   - Use CLAHE for local contrast enhancement
   - Apply sharpening for edge emphasis

### Step 3: Train the Model

1. **Capture Good Samples**:
   - Place defect-free bottles under camera
   - Click "Capture Good" for each bottle
   - Collect 20+ samples per variant

2. **Select Bottle Variant**:
   - Choose from 12ml/8ml, black/white variants
   - Each variant needs separate training

3. **Train Model**:
   - Click "Train" button
   - Wait for "Training completed" message
   - Model saves automatically to `models/` folder

### Step 4: Test and Validate

1. **Single Test**:
   - Place a bottle in view
   - Click "Test Once"
   - Check score and decision

2. **Adjust Threshold**:
   - Start with default 3.5
   - Increase for stricter inspection
   - Decrease for more lenient inspection

### Step 5: Production Inspection

1. **Start Inspection**:
   - Click "Start Inspection"
   - System begins monitoring sensor input

2. **Auto Mode** (Optional):
   - Set target count for automatic capture
   - Useful for collecting large datasets

3. **Monitor Results**:
   - Watch "Result" display for OK/NG decisions
   - Check status messages for system state

## ⚙️ Configuration Parameters

### Critical Timing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Pre-activation | 0ms | Delay before main solenoid |
| Capture Offset | 40ms | Camera capture timing |
| Main Active | 220ms | Main solenoid hold time |
| Reject After | 50ms | Delay before reject action |
| Reject Active | 180ms | Reject solenoid pulse time |

### Model Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| Threshold | 3.5 | Sensitivity to defects |
| Inference Scale | 160px | Processing resolution |
| Embedding Dims | 80 | Feature dimensions |

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera connection
   - Verify `CAM_INDEX` in code
   - Try different USB port

2. **Model Training Fails**
   - Ensure 20+ good samples
   - Check ROI covers bottle area
   - Verify write permissions

3. **Poor Detection Accuracy**
   - Collect more training samples
   - Adjust enhancement parameters
   - Fine-tune threshold value

4. **PLC Communication Errors**
   - Verify COM port settings
   - Check baud rate (default: 9600)
   - Confirm Modbus address mapping

### Performance Optimization

- **For Faster Inference**: Reduce `INFER_DOWNSCALE` to 128
- **For Better Accuracy**: Increase training samples to 50+
- **For CPU Systems**: Reduce `EMBED_DIMS` to 50

## 📁 Project Structure

```
bottle_inspector/
├── main.py                 # Application entry point
├── inspector_app.py        # Main GUI and logic
├── resnet_feature_extractor.py  # Deep learning model
├── requirements.txt        # Dependencies
├── data_variants/          # Training images
│   ├── 12ml_black/
│   ├── 12ml_white/
│   ├── 8ml_black/
│   └── 8ml_white/
├── models/                 # Trained models
│   ├── padim_12ml_black.npz
│   └── ...
└── roi_meta.json          # ROI configuration
```

## 🛠️ Advanced Features

### Multi-Variant Support
- Train separate models for different bottle types
- Automatic model switching during inspection
- Independent threshold per variant

### Image Enhancement Pipeline
- CLAHE for contrast enhancement
- Gamma correction for brightness
- Unsharp masking for edge enhancement
- Real-time preview of enhancements

### Industrial Integration
- Modbus RTU communication
- Ref-counted solenoid control
- Event queuing for high-speed operation
- Configurable timing parameters

## 📊 Expected Performance

- **Inference Time**: 50-100ms (depending on hardware)
- **Accuracy**: >95% with proper training
- **Throughput**: 5-10 bottles/second
- **Minimum Training**: 20 samples per variant

## 🔒 Quality Assurance

For reliable crack detection:

1. **Consistent Lighting**: Maintain stable illumination
2. **Proper Focus**: Ensure camera is properly focused
3. **Adequate Training**: Use representative good samples
4. **Regular Validation**: Periodically test with known defects
5. **Environmental Control**: Minimize background variations

## 🎥 Demo Video

*[Link to demonstration video would be placed here]*



---

**Note**: This system provides industrial-grade crack detection capabilities that can be adapted to various bottle types and production environments. Proper setup and training are crucial for optimal performance.

*Last Updated: 2024*
