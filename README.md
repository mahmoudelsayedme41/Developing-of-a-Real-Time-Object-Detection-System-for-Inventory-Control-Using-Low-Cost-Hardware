# Real-Time Animal Feed Bag Counter

An advanced computer vision system for real-time detection and counting of different types of animal feed bags using YOLOv8. The system supports both PC and Raspberry Pi deployments with robust tracking, performance monitoring, and automated Excel reporting.

## 🎯 Features

- **Real-time Detection**: YOLOv8-based object detection for 12 different feed bag types
- **Multi-platform Support**: Optimized versions for both PC and Raspberry Pi
- **Advanced Tracking**: Enhanced object tracking with confidence smoothing
- **Automated Reporting**: Excel export with Arabic labels and performance statistics
- **Performance Monitoring**: Real-time FPS, processing time, and system resource tracking
- **Robust Camera Handling**: Multiple backend support with automatic fallback
- **Background Images**: Synthetic background generation for better model training

## 📋 Detected Classes

The system can detect and count the following animal feed types:

1. 14% روا د بياض دواجن (14% Laying Hen Feed - White)
2. 14% روا د تسمين مواشي (14% Cattle Fattening Feed)
3. 16% روا د حلا ب مواشي (16% Dairy Cattle Feed)
4. 16% روا د بياض دواجن (16% Laying Hen Feed - White)
5. 16% روا د تسمين مواشي (16% Cattle Fattening Feed)
6. 19% روا د حلا ب عالي الإدار مواشي (19% High-Performance Dairy Feed)
7. 19% روا د سوبر دواجن (19% Super Poultry Feed)
8. 20% روا د فطام بتلو مواشي (20% Calf Weaning Feed)
9. 21% روا د سوبر دواجن (21% Super Poultry Feed)
10. 21% روا د بادي نامي محبوب دواجن (21% Preferred Growing Poultry Feed)
11. 21% روا د بادي نامي مفتت دواجن (21% Crumbled Growing Poultry Feed)
12. 23% روا د سوبر دواجن (23% Super Poultry Feed)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install ultralytics opencv-python numpy xlsxwriter albumentations psutil
```

### For PC Deployment

```bash
python pc_bag_counter.py
```

### For Raspberry Pi Deployment

```bash
python realtime_bag_counter_pi.py
```

### Training Mode (Google Colab)

Use the provided Jupyter notebook `training_notebook.ipynb` for model training with data augmentation.

## 🏗️ Project Structure

```
bag-counter/
├── models/
│   └── best.pt                    # Trained YOLOv8 model
├── src/
│   ├── pc_bag_counter.py          # PC optimized version
│   ├── realtime_bag_counter_pi.py # Raspberry Pi version
│   └── training_notebook.ipynb    # Training pipeline
├── data/
│   ├── images/                    # Training images
│   ├── labels/                    # YOLO format labels
│   └── classes.txt               # Class names
├── output/
│   └── bag_counts/               # Excel reports
├── docs/
│   ├── setup.md                  # Detailed setup guide
│   ├── api_reference.md          # Code documentation
│   └── troubleshooting.md        # Common issues
├── requirements.txt
├── config.yaml                   # Configuration file
└── README.md
```

## ⚙️ Configuration

### PC Configuration

```python
# File paths - CHANGE THESE FOR YOUR SETUP
MODEL_PATH = r"D:\ready\bag_counter\yolo_model\best.pt"
CAMERA_INDEX = 0

# Video settings optimized for PC
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
```

### Raspberry Pi Configuration

```python
# Optimized for Raspberry Pi 5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
SKIP_FRAMES = 2  # Process every nth frame
```

## 🎮 Controls

- **'q'**: Quit application
- **'s'**: Save current counts manually
- **'r'**: Reset all counts
- **'f'**: Toggle fullscreen (PC only)
- **ESC**: Emergency quit

## 📊 Performance

### PC Performance
- **FPS**: 30+ (1280x720)
- **Processing Time**: 15-30ms per frame
- **GPU Support**: CUDA acceleration when available
- **Multi-threading**: Optimized for multiple CPU cores

### Raspberry Pi Performance
- **FPS**: 10-15 (640x480)
- **Processing Time**: 100-200ms per frame
- **Frame Skipping**: Every 2nd frame for real-time performance
- **Memory Optimized**: Low memory footprint

## 📈 Data Pipeline

### 1. Data Collection
Real factory images collected and labeled using Label Studio

### 2. Data Augmentation
```python
# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.Blur(p=0.2),
    A.HueSaturationValue(p=0.3),
])
```

### 3. Model Training
- **Base Model**: YOLOv8m
- **Epochs**: 200
- **Image Size**: 640x640
- **Batch Size**: 16
- **Data Split**: 90% train, 10% validation

## 📋 Output Format

The system generates Excel reports with:

- **Arabic Headers**: Native Arabic text support
- **Per-Class Counts**: Individual counts for each feed type
- **Confidence Scores**: Average detection confidence
- **Timestamps**: When each detection occurred
- **Performance Metrics**: System performance statistics
- **Bag IDs**: Unique tracking identifiers

## 🛠️ Hardware Requirements

### PC Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GTX 1060 or better (optional but recommended)
- **Camera**: USB 2.0+ compatible camera
- **Storage**: 2GB free space

### Raspberry Pi Requirements
- **Model**: Raspberry Pi 4B or 5 (8GB RAM recommended)
- **Camera**: Pi Camera Module or USB camera
- **Storage**: 32GB+ microSD card (Class 10)
- **Cooling**: Active cooling recommended for continuous operation

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```bash
   python pc_bag_counter.py --diagnostics
   ```

2. **Model Loading Errors**
   - Verify model path in configuration
   - Check PyTorch compatibility

3. **Performance Issues**
   - Reduce frame size
   - Increase SKIP_FRAMES value
   - Enable GPU acceleration

## 📚 API Reference

### Key Classes

- `PCRealTimeBagCounter`: Main application for PC
- `RealTimeBagCounter`: Raspberry Pi optimized version
- `EnhancedTracker`: Object tracking with confidence smoothing
- `ExcelSaver`: Automated Excel report generation
- `PerformanceMonitor`: Real-time performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Factory team for providing real-world data
- Label Studio for annotation tools
- Ultralytics for YOLOv8 framework
- OpenCV community for computer vision tools

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the troubleshooting guide
- Review the API documentation

---

**Built with ❤️ for industrial automation and quality control**