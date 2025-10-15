#!/usr/bin/env python3
"""
UV-based setup script for ImageNet training project
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        
        if result.returncode == 0:
            print(f"✅ {description} - Success")
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
        
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed with error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - Command not found")
        return False


def check_uv_installed():
    """Check if UV is installed"""
    print("🔍 Checking UV installation...")
    
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ UV is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ UV is not working properly")
            return False
    except FileNotFoundError:
        print("❌ UV is not installed")
        return False


def install_uv():
    """Install UV package manager"""
    print("\n📦 Installing UV...")
    
    if sys.platform == "win32":
        # Windows installation
        cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
        success = run_command(cmd, "Installing UV on Windows", check=False)
    else:
        # macOS/Linux installation
        cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        success = run_command(cmd, "Installing UV on macOS/Linux", check=False)
    
    if success:
        print("✅ UV installation completed")
        print("⚠️  You may need to restart your terminal or source your shell profile")
        return True
    else:
        print("❌ UV installation failed")
        print("💡 Please install manually: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def create_venv():
    """Create virtual environment with UV"""
    print("\n🐍 Creating virtual environment...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print(f"⚠️  Virtual environment already exists at {venv_path}")
        return True
    
    # Try different Python versions
    python_versions = ["3.11", "3.10", "3.9", "3.8", "python3", "python"]
    
    for py_version in python_versions:
        cmd = f"uv venv --python {py_version}"
        if run_command(cmd, f"Creating venv with {py_version}", check=False):
            return True
    
    print("❌ Failed to create virtual environment with any Python version")
    return False


def install_dependencies():
    """Install project dependencies"""
    print("\n📚 Installing dependencies...")
    
    # Install in editable mode with dev dependencies
    cmd = "uv pip install -e ."
    if not run_command(cmd, "Installing project in editable mode", check=False):
        # Fallback to sync from pyproject.toml
        cmd = "uv pip sync pyproject.toml"
        if not run_command(cmd, "Installing dependencies from pyproject.toml", check=False):
            return False
    
    # Install dev dependencies
    cmd = "uv pip install -e .[dev]"
    run_command(cmd, "Installing development dependencies", check=False)
    
    return True


def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    # List installed packages
    if run_command("uv pip list", "Listing installed packages", check=False):
        pass
    
    # Test import of key packages
    test_imports = [
        "torch",
        "torchvision", 
        "tqdm",
        "PIL",
        "numpy"
    ]
    
    print("\n🧪 Testing imports...")
    all_good = True
    
    for package in test_imports:
        try:
            cmd = f"python -c \"import {package}; print(f'{package} version: {{getattr({package}, '__version__', 'unknown')}})\""
            if run_command(cmd, f"Testing {package} import", check=False):
                pass
            else:
                all_good = False
        except:
            print(f"❌ Failed to test {package}")
            all_good = False
    
    return all_good


def main():
    """Main setup function"""
    print("🚀 UV-based ImageNet Training Setup")
    print("=" * 50)
    
    # Change to parent directory (project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"📁 Working directory: {project_root}")
    
    # Check current directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Are you in the right directory?")
        sys.exit(1)
    
    # Step 1: Check/Install UV
    if not check_uv_installed():
        if not install_uv():
            print("\n❌ Setup failed: UV installation unsuccessful")
            sys.exit(1)
        
        # Check again after installation
        if not check_uv_installed():
            print("\n❌ Setup failed: UV still not available after installation")
            print("💡 Try restarting your terminal and running this script again")
            sys.exit(1)
    
    # Step 2: Create virtual environment
    if not create_venv():
        print("\n❌ Setup failed: Could not create virtual environment")
        sys.exit(1)
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Step 4: Verify installation
    verify_installation()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    
    if sys.platform == "win32":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Test the setup:")
    print("   python test_setup.py")
    
    print("\n3. Start training (with ImageNet dataset):")
    print("   python train_imagenet.py --data-dir /path/to/imagenet")
    
    print("\n📚 Documentation:")
    print("   - UV Guide: README_uv.md")
    print("   - ImageNet Training: README_imagenet.md")


if __name__ == "__main__":
    main()