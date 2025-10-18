# ImageNet Dataset Setup and X11 Forwarding Guide

## 📁 Dataset Location
The ImageNet 1000 dataset has been downloaded from Kaggle and is available at:
```
/home/ubuntu/Downloads/
```

## 🖥️ X11 Forwarding Setup with VcXsrv

This guide helps you set up X11 forwarding to run GUI applications (like Chrome) from your EC2 instance on your Windows machine.

### Prerequisites
- Windows 10/11 machine
- EC2 instance with SSH access
- VcXsrv installed on Windows

---

## 🚀 Step-by-Step Setup

### 1️⃣ Verify SSH Client Compatibility

Open **PowerShell** or **Command Prompt** on Windows and check your SSH version:

```bash
ssh -V
```

Expected output:
```
OpenSSH_for_Windows_8.x or newer
```

If you see an older version, use Windows' built-in SSH client:
```
C:\Windows\System32\OpenSSH\ssh.exe
```

### 2️⃣ Install and Configure VcXsrv

#### Download and Install
1. Download VcXsrv from: https://sourceforge.net/projects/vcxsrv/
2. Run installer with default settings

#### Launch VcXsrv (XLaunch)
1. Start **XLaunch** from Start Menu
2. Configuration wizard:
   - **Display settings**: Select "Multiple windows" → Next
   - **Client startup**: Select "Start no client" → Next
   - **Extra settings**: 
     - ✅ Check "Clipboard"
     - ✅ Check "Primary Selection" 
     - ✅ Check "Native opengl"
     - ✅ Check "Disable access control" (Important!)
   - **Finish**: Save configuration for future use

3. Verify VcXsrv is running - you should see a white "X" icon in your system tray

### 3️⃣ Configure Windows Firewall

When VcXsrv starts, Windows may ask for firewall access:
- **Allow access** for both private and public networks
- If not prompted, manually add firewall rule for VcXsrv

### 4️⃣ Set Up SSH Configuration

#### Method A: SSH Config File (Recommended)

Create or edit the SSH config file on your Windows machine:
```
C:\Users\<your-username>\.ssh\config
```

Add the following configuration:
```
Host my-ec2
    HostName ec2-3-10-226-102.eu-west-2.compute.amazonaws.com
    User ubuntu
    IdentityFile C:\path\to\your\key.pem
    ForwardX11 yes
    ForwardX11Trusted yes
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

Then connect using:
```bash
ssh my-ec2
```

#### Method B: Command Line Flags

Connect directly with X11 forwarding flags:
```bash
ssh -Y ubuntu@ec2-3-10-226-102.eu-west-2.compute.amazonaws.com -i path\to\your\key.pem
```

### 5️⃣ Verify X11 Forwarding

After connecting to EC2, check if X11 forwarding is working:

```bash
# Check DISPLAY variable
echo $DISPLAY
# Expected output: localhost:10.0 or similar

# Test with a simple X11 application
sudo apt install -y x11-apps
xeyes
```

If `xeyes` opens a window on your Windows desktop, X11 forwarding is working!

### 6️⃣ Alternative: Using PuTTY

If PowerShell SSH doesn't work, try PuTTY:

1. **Download PuTTY**: https://www.putty.org/
2. **Convert your .pem key to .ppk** using PuTTYgen
3. **Configure PuTTY**:
   - Session → Hostname: `ubuntu@your-ec2-address`
   - Connection → SSH → Auth → Browse to your .ppk key
   - Connection → SSH → X11 → ✅ Enable X11 forwarding
   - X display location: `localhost:0`
4. **Save** the session and **Open**

---

## 🌐 Running Chrome with GUI

Once X11 forwarding is set up, you can run Chrome with GUI:

```bash
# Launch Chrome (will open on your Windows desktop)
google-chrome-stable --no-sandbox

# Or in headless mode for automation
google-chrome-stable --headless --no-sandbox --remote-debugging-port=9222
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Error: Can't open display"
```bash
# Check if DISPLAY is set
echo $DISPLAY

# If empty, try setting manually for testing
export DISPLAY=localhost:0.0
xeyes
```

#### VcXsrv not receiving connections
1. Ensure "Disable access control" is checked in VcXsrv
2. Check Windows Firewall settings
3. Restart VcXsrv and reconnect SSH

#### SSH not forwarding X11
1. Verify SSH client supports X11 forwarding
2. Check SSH server config on EC2:
   ```bash
   grep X11Forwarding /etc/ssh/sshd_config
   # Should show: X11Forwarding yes
   ```
3. Ensure xauth is installed:
   ```bash
   sudo apt install -y xauth
   ```

#### Performance Issues
1. Add compression to SSH:
   ```bash
   ssh -Y -C ubuntu@your-ec2-address
   ```
2. Use local Chrome for better performance when possible

---

## 📊 Dataset Information

### ImageNet 1000 Classes Dataset
- **Location**: `/home/ubuntu/Downloads/`
- **Source**: Kaggle
- **Classes**: 1000 object categories
- **Format**: Standard ImageNet directory structure

### Directory Structure
```
/home/ubuntu/Downloads/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
```

---

## 🔄 Quick Start Commands

### Start VcXsrv Session
1. Launch XLaunch on Windows
2. Connect with SSH:
   ```bash
   ssh -Y ubuntu@your-ec2-address
   ```
3. Test X11:
   ```bash
   xeyes
   ```
4. Launch Chrome:
   ```bash
   google-chrome-stable --no-sandbox
   ```

### VS Code Integration
You can also use VS Code's Simple Browser for web interfaces:
- Open Command Palette (Ctrl+Shift+P)
- Type "Simple Browser"
- Navigate to localhost interfaces

---

## 📝 Notes

- **Security**: Only use "Disable access control" in VcXsrv for trusted networks
- **Performance**: X11 forwarding adds latency - use for setup/debugging, not production
- **Persistence**: VcXsrv settings can be saved for automatic startup
- **Multiple Sessions**: Each SSH connection gets its own DISPLAY number

---

## 📚 Additional Resources

- [VcXsrv Documentation](https://sourceforge.net/projects/vcxsrv/)
- [SSH X11 Forwarding Guide](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)
- [ImageNet Dataset](https://www.image-net.org/)

---

*Last updated: October 18, 2025*
*Environment: EC2 Ubuntu 24.04, Windows 10/11 with VcXsrv*