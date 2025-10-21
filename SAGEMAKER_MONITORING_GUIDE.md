# 🔍 SageMaker Job Monitoring Guide

## Current Situation
Your SageMaker job submission is hanging at the data input stage. This typically happens during the `estimator.fit()` call when submitting to AWS.

## 🚀 Immediate Actions

### 1. Check if Job is Running in Background

**Option A: Use the monitoring script**
```bash
# Check recent jobs
python monitor_sagemaker_jobs.py

# Monitor specific job in real-time
python monitor_sagemaker_jobs.py --job-name "your-job-name" --monitor

# Search for jobs with pattern
python monitor_sagemaker_jobs.py --pattern "imagenet"
```

**Option B: AWS CLI**
```bash
# List recent training jobs
aws sagemaker list-training-jobs --max-results 10 --sort-by CreationTime --sort-order Descending

# Check specific job status
aws sagemaker describe-training-job --training-job-name "your-job-name"
```

**Option C: AWS Console**
- Open: https://console.aws.amazon.com/sagemaker/home#/jobs
- Look for your job in the training jobs list

### 2. Get Real-time Logs

**CloudWatch Logs:**
```bash
# List log groups
aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker/TrainingJobs"

# Get log streams for your job
aws logs describe-log-streams --log-group-name "/aws/sagemaker/TrainingJobs" --log-stream-name-prefix "your-job-name"

# Stream logs in real-time
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-names "your-job-name/algo-1-{timestamp}"
```

### 3. Check Why It's Hanging

**Common reasons for hanging at data input stage:**

1. **AWS API Throttling** - AWS is rate limiting your requests
2. **Large Instance Provisioning** - p3.8xlarge + spot can take 10+ minutes
3. **Authentication Issues** - Role permissions or credentials
4. **S3 Access Problems** - Bucket permissions or data availability
5. **Service Issues** - AWS SageMaker service degradation

## 🛠️ Troubleshooting Steps

### Step 1: Verify AWS Setup
```bash
# Check credentials
aws sts get-caller-identity

# Check region
aws configure get region

# Test S3 access
aws s3 ls s3://tsai-era-v4-mini-capstone/Datasets/imagenet1k/ILSVRC/
```

### Step 2: Check Service Health
- AWS Service Health Dashboard: https://status.aws.amazon.com/
- Look for SageMaker or EC2 issues in your region

### Step 3: Monitor System Resources
```bash
# Check if your process is still running
ps aux | grep python

# Check network activity
netstat -an | grep ESTABLISHED
```

## 📊 Expected Timeline

**Normal SageMaker job submission:**
- ⏳ **0-2 minutes**: API validation and job creation
- 🏗️ **2-5 minutes**: Instance provisioning (longer for spot)
- 📥 **5-10 minutes**: Data download from S3
- 🚀 **10+ minutes**: Training starts

**For p3.8xlarge spot instances:**
- Can take **15-30 minutes** just for provisioning
- AWS may retry multiple availability zones

## 🎯 What to Do Right Now

1. **Don't kill the process yet** - It might be working in background
2. **Open another terminal** and run:
   ```bash
   python monitor_sagemaker_jobs.py
   ```
3. **Check AWS Console** to see if job appears
4. **Wait 30 minutes total** before considering timeout
5. **Check CloudWatch logs** if job appears but hangs

## 🚨 If Still Hanging After 30 Minutes

**Force timeout and retry with smaller instance:**
```bash
# Kill current process (Ctrl+C)
# Then retry with smaller instance:
python sagemaker_orchestrator.py \
  --role-arn "your-role-arn" \
  --source-bucket "tsai-era-v4-mini-capstone" \
  --use-spot \
  --epochs 1 \
  --instance-type "ml.g5.4xlarge"  # Smaller instance
```

## 📋 Log Analysis

**If job shows up in AWS Console:**
- Status "Starting" = Normal, instance provisioning
- Status "InProgress" + "Downloading" = Downloading data
- Status "InProgress" + "Training" = Actually training
- Status "Failed" = Check failure reason

**Key log patterns to look for:**
```
Starting - The training job is starting
Downloading - Training data is being downloaded
Training - Model training has begun
Uploading - Training completed, uploading results
Completed - Job finished successfully
```

## 🔧 Enhanced Monitoring

The updated code now includes:
- ✅ 5-minute timeout for job submission
- ✅ Real-time status checking
- ✅ Better error messages
- ✅ Progress indicators
- ✅ AWS API response monitoring

Your next run will show exactly where it hangs!