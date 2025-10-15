# Setup Tools

This folder contains all setup and configuration scripts for the ImageNet training project.

## 📁 Contents

- `setup_uv.py` - Comprehensive UV-based environment setup
- `test_setup.py` - Verify that all dependencies and components work correctly
- `quick_setup.py` - Configure training files for downloaded datasets
- `README.md` - This documentation file

## 🚀 Quick Start

### 1. **Complete Environment Setup**
```bash
cd setup
python setup_uv.py
```
This will:
- ✅ Install UV package manager (if needed)
- ✅ Create virtual environment
- ✅ Install all dependencies from pyproject.toml
- ✅ Verify the installation

### 2. **Test Your Setup**
```bash
python test_setup.py
```
This will verify:
- ✅ Model creation and forward pass
- ✅ Data transforms functionality
- ✅ CUDA availability
- ✅ All imports work correctly

### 3. **Configure for Dataset**
```bash
python quick_setup.py
```
This will:
- ✅ Detect available datasets
- ✅ Update training file configurations
- ✅ Generate dataset-specific settings
- ✅ Create test scripts with correct paths

## 🔧 Individual Tools

### `setup_uv.py` - Environment Setup
**Purpose**: One-command setup for the entire development environment

**Features**:
- Cross-platform UV installation
- Virtual environment creation
- Dependency installation from pyproject.toml
- Comprehensive error handling and verification

**Usage**:
```bash
python setup_uv.py
```

### `test_setup.py` - Verification
**Purpose**: Comprehensive testing of the installed environment

**Tests**:
- Model architecture (ResNet50)
- PyTorch functionality
- Data transforms
- GPU/CUDA availability
- Import system

**Usage**:
```bash
python test_setup.py
```

### `quick_setup.py` - Dataset Configuration
**Purpose**: Configure training files for specific datasets

**Features**:
- Auto-detect downloaded datasets
- Generate dataset-specific configurations
- Create test scripts with correct parameters
- Update training file paths

**Usage**:
```bash
python quick_setup.py
```

## 🎯 Workflow

### First-time Setup
1. **Environment**: `python setup_uv.py`
2. **Verify**: `python test_setup.py`
3. **Download data**: `cd ../dataset_tools` → use downloader notebook
4. **Configure**: `python quick_setup.py`
5. **Start training**: `cd .. && python train_imagenet.py`

### Regular Usage
- **Add dependencies**: Update `../pyproject.toml` → run `python setup_uv.py`
- **Verify after changes**: `python test_setup.py`
- **New dataset**: Use dataset tools → `python quick_setup.py`

## 💡 Tips

### Environment Management
- All scripts automatically work from the correct project directory
- Virtual environment is created in project root as `.venv/`
- Dependencies are managed through `pyproject.toml`

### Troubleshooting
- **Import errors**: Run `python test_setup.py` to diagnose
- **UV issues**: Check UV installation with `uv --version`
- **Path problems**: Ensure you're running from the `setup/` folder

### Cross-Platform Notes
- Works on Windows, macOS, and Linux
- UV installation is automatic for all platforms
- PowerShell, Bash, and Zsh are all supported

## 🔗 Integration

These setup tools work seamlessly with:
- `../dataset_tools/` - Dataset download and management
- `../lr_optimization/` - Learning rate optimization
- `../data_exploration/` - Dataset visualization
- Main training scripts in project root

## ⚡ Performance

### Setup Times
- **First run**: 2-5 minutes (includes UV installation)
- **Subsequent runs**: 30-60 seconds (cached dependencies)
- **Verification**: 10-30 seconds

### Resource Usage
- **Disk space**: ~2GB for full environment
- **Memory**: Minimal during setup, 1-8GB during training
- **Network**: Downloads dependencies once, cached locally

## 🎉 Success Indicators

### ✅ Good Setup
- All commands complete without errors
- Test script shows all green checkmarks
- GPU detected (if available)
- Quick training test runs successfully

### ❌ Common Issues
- **UV not found**: Restart terminal after setup
- **Import errors**: Check virtual environment activation
- **CUDA issues**: Verify GPU drivers and PyTorch installation
- **Path errors**: Ensure running from correct directory

After successful setup, you're ready to train ResNet50 on ImageNet! 🚀