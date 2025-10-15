# Dataset Tools

This folder contains tools for downloading and managing ImageNet subset datasets for your ResNet50 training experiments.

## 📁 Contents

- `imagenet_subset_downloader.ipynb` - Interactive Jupyter notebook for downloading datasets

## 🚀 Quick Start

1. **Open the notebook:**
   ```bash
   jupyter notebook imagenet_subset_downloader.ipynb
   ```

2. **Select and download a dataset** using the interactive widgets

3. **Go back to main folder** and start training:
   ```bash
   cd ..
   python quick_test.py
   ```

## 📊 Available Datasets

### 🔹 Tiny ImageNet (Recommended for first try)
- **Size**: ~240MB
- **Classes**: 200 from ImageNet
- **Image size**: 64x64 pixels
- **Training time**: 10-30 minutes
- **Perfect for**: Quick experiments and testing

### 🔸 Imagenette (Fast experiments)
- **Size**: ~300MB  
- **Classes**: 10 from ImageNet
- **Image size**: 320x320 pixels
- **Training time**: 5-15 minutes
- **Perfect for**: Rapid prototyping

### 🔹 ImageWoof (Challenging task)
- **Size**: ~300MB
- **Classes**: 10 dog breeds
- **Image size**: 320x320 pixels  
- **Training time**: 5-15 minutes
- **Perfect for**: Testing on challenging similar classes

## 🎯 Features

- **Interactive download** with progress bars
- **Automatic dataset organization** (especially for Tiny ImageNet validation data)
- **Training command generation** with optimal parameters
- **Dataset verification** and status checking
- **Error handling** and resume capability

## 💡 Tips

- Start with **Tiny ImageNet** for your first experiment
- Use the **learning rate finder** before full training
- Try **pretrained weights** with Imagenette/ImageWoof for better results
- Monitor initial epochs closely to catch issues early

## 🔗 Integration

The notebook automatically places datasets in the `../datasets/` folder, making them ready to use with:
- `train_imagenet.py` - Main training script
- `lr_optimization/learning_rate_finder.ipynb` - LR optimization
- `quick_test.py` - Setup verification

Happy experimenting! 🚀