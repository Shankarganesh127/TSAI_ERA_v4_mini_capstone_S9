@echo off
REM SageMaker ImageNet Training Launcher (Windows)
REM This batch script provides easy access to SageMaker training functionality

echo ====================================
echo   SageMaker ImageNet Training
echo   TSAI ERAv4 Mini Capstone S9  
echo ====================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Navigate to sagemaker_training directory
cd /d "%~dp0"

REM Check if required modules are installed
python -c "import sagemaker" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        exit /b 1
    )
)

echo.
echo Available commands:
echo 1. Launch SageMaker Training Job
echo 2. Upload ImageNet Dataset to S3
echo 3. Quick Test Training
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto launch_training
if "%choice%"=="2" goto upload_data
if "%choice%"=="3" goto test_training
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto menu

:launch_training
echo.
echo === SageMaker Training Job Launcher ===
set /p s3_data="Enter S3 data URI (e.g., s3://your-bucket/imagenet-1k/): "
if "%s3_data%"=="" (
    echo ERROR: S3 data URI is required
    goto menu
)

set /p instance_type="Enter instance type [ml.g4dn.xlarge]: "
if "%instance_type%"=="" set instance_type=ml.g4dn.xlarge

set /p epochs="Enter number of epochs [90]: "
if "%epochs%"=="" set epochs=90

set /p batch_size="Enter batch size [256]: "
if "%batch_size%"=="" set batch_size=256

echo.
echo Launching training with:
echo - Data: %s3_data%
echo - Instance: %instance_type%
echo - Epochs: %epochs%
echo - Batch size: %batch_size%
echo.

python launch_sagemaker_job.py ^
  --train-data-s3 "%s3_data%" ^
  --instance-type "%instance_type%" ^
  --epochs %epochs% ^
  --batch-size %batch_size%

goto end

:upload_data
echo.
echo === ImageNet Dataset S3 Upload ===
set /p local_path="Enter local ImageNet dataset path: "
if "%local_path%"=="" (
    echo ERROR: Local dataset path is required
    goto menu
)

set /p bucket_name="Enter S3 bucket name: "
if "%bucket_name%"=="" (
    echo ERROR: S3 bucket name is required
    goto menu
)

set /p s3_prefix="Enter S3 prefix [imagenet-1k]: "
if "%s3_prefix%"=="" set s3_prefix=imagenet-1k

echo.
echo Uploading dataset:
echo - Local path: %local_path%
echo - S3 bucket: %bucket_name%
echo - S3 prefix: %s3_prefix%
echo.

python upload_imagenet_to_s3.py ^
  --local-data "%local_path%" ^
  --bucket "%bucket_name%" ^
  --s3-prefix "%s3_prefix%" ^
  --create-bucket

goto end

:test_training
echo.
echo === Quick Test Training ===
set /p s3_data="Enter S3 data URI for testing: "
if "%s3_data%"=="" (
    echo ERROR: S3 data URI is required
    goto menu
)

echo.
echo Launching quick test training (1 epoch, small batch)...

python launch_sagemaker_job.py ^
  --train-data-s3 "%s3_data%" ^
  --instance-type ml.g4dn.xlarge ^
  --epochs 1 ^
  --batch-size 64 ^
  --quick-mode ^
  --job-name imagenet-test

goto end

:end
echo.
echo Training job submitted. Check AWS SageMaker console for status.
pause

:exit
exit /b 0