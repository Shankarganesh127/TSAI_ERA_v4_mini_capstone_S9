# ImageNet-1K Dataset Exploration 🔍

This folder contains tools for comprehensive exploration and visualization of the ImageNet-1K dataset.

## 📁 Files

- **`imagenet_dataset_explorer.ipynb`** - Main Jupyter notebook with interactive exploration tools
- **`README.md`** - This documentation file

## 🚀 Quick Start

### 1. Setup Dependencies

```bash
# Install required packages with UV
uv pip install matplotlib pillow numpy pandas seaborn ipywidgets

# Or verify setup from main project folder
cd ../setup && python test_setup.py
```

### 2. Start Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab
jupyter lab
```

### 3. Open the Explorer

1. Navigate to `imagenet_dataset_explorer.ipynb`
2. Update the `IMAGENET_ROOT` path in cell 2
3. Run all cells to start exploring

## 🎯 Features

### 📊 Dataset Analysis
- **Structure browsing** - Explore folder organization and file counts
- **Class distribution** - Visualize image counts across classes
- **Metadata analysis** - File sizes, dimensions, formats

### 🖼️ Image Exploration
- **Sample viewing** - Display random images from any class
- **Interactive browser** - Widget-based image navigation
- **Detailed information** - File properties and image statistics

### 🎨 Visual Analytics
- **Color analysis** - RGB channel distributions and histograms
- **Property analysis** - Brightness, contrast, aspect ratios
- **Comparative visualization** - Compare classes and splits

### 🎛️ Interactive Tools
- **Class selector** - Dropdown to choose classes
- **Navigation controls** - Previous, Next, Random buttons
- **Split selection** - Toggle between train/validation sets
- **Real-time info** - Dynamic image and file information display

## 📋 Requirements

### Essential
- Python 3.8+
- matplotlib (plotting)
- PIL/Pillow (image processing)
- numpy (numerical computing)
- pandas (data analysis)

### Optional
- ipywidgets (interactive features)
- seaborn (enhanced plotting)
- jupyter (notebook environment)

## 📂 Dataset Setup

Your ImageNet dataset should be organized as:

```
imagenet/
├── train/
│   ├── n01440764/    # Class folder (WordNet ID)
│   │   ├── image1.JPEG
│   │   ├── image2.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...           # 1000 class folders
└── val/
    ├── n01440764/
    │   ├── image1.JPEG
    │   └── ...
    └── ...
```

## 🎮 Usage Examples

### Basic Exploration
```python
# Display 6 random images from a class
display_sample_images('n01440764', num_images=6, split='train')

# Analyze image metadata
metadata = analyze_image_metadata('n01440764', max_images=50)
visualize_metadata(metadata, 'n01440764')
```

### Advanced Analysis
```python
# Analyze color properties
analyze_image_properties('n01440764', max_images=100)

# Visualize class distribution
visualize_class_distribution(max_classes=50)
```

### Interactive Exploration
Run the interactive explorer cell to get:
- Class selection dropdown
- Image navigation controls
- Real-time image information
- Seamless browsing experience

## 🔧 Customization

You can modify the notebook for your specific needs:

### Adjust Sample Sizes
```python
MAX_IMAGES_PER_CLASS = 20  # Increase for more samples
FIGURE_SIZE = (20, 15)     # Larger plots
```

### Add Custom Analysis
```python
def custom_analysis(class_name):
    # Your custom analysis code here
    pass
```

### Filter Classes
```python
# Analyze only specific classes
target_classes = ['n01440764', 'n01443537', 'n01484850']
for class_name in target_classes:
    analyze_image_properties(class_name)
```

## 💡 Tips

1. **Start Small** - Begin with a few classes to test your setup
2. **Update Paths** - Ensure the `IMAGENET_ROOT` path is correct
3. **Use Interactive Mode** - Leverage widgets for efficient browsing
4. **Save Insights** - Document interesting findings in new cells
5. **Compare Classes** - Run the same analysis on different classes
6. **Monitor Performance** - Large sample sizes may take time to process

## 🐛 Troubleshooting

### Common Issues

**Notebook not starting:**
```bash
pip install jupyter
jupyter notebook --version
```

**Images not loading:**
- Check the `IMAGENET_ROOT` path
- Verify dataset folder structure
- Ensure image files have correct extensions

**Interactive widgets not working:**
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

**Slow performance:**
- Reduce `max_images` parameters
- Analyze fewer classes at once
- Use smaller figure sizes

### Getting Help

1. Check the main project README for general setup
2. Run `cd ../setup && python test_setup.py` to verify dependencies
3. Use the organized subfolder structure for easy navigation

## 🎉 Next Steps

After exploring your dataset:

1. **Identify interesting classes** for detailed analysis
2. **Compare train/val distributions** for data quality assessment
3. **Document findings** for preprocessing decisions
4. **Select representative samples** for model testing
5. **Use insights** to inform training strategy

Happy exploring! 🚀