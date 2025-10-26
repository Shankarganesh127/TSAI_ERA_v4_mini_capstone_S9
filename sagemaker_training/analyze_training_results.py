import os
import pandas as pd
import matplotlib.pyplot as plt
import json

OUTPUT_DIR = './imagenet_pipeline_results'

# --- Training Log Analysis ---
train_log_csv = os.path.join(OUTPUT_DIR, 'train_log.csv')
if os.path.exists(train_log_csv):
    df = pd.read_csv(train_log_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curve.png'))
    plt.close()
else:
    print('No train_log.csv found.')

# --- LR Finder Analysis ---
lr_curve_csv = os.path.join(OUTPUT_DIR, 'lr_range_curve.csv')
if os.path.exists(lr_curve_csv):
    df = pd.read_csv(lr_curve_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(df['lr'], df['smoothed_loss'])
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Smoothed Loss')
    plt.title('LR Range Test Curve')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'lr_range_curve_plot.png'))
    plt.close()
else:
    print('No lr_range_curve.csv found.')

# --- Weight Decay Search Analysis ---
wd_csv = os.path.join(OUTPUT_DIR, 'weight_decay_search.csv')
if os.path.exists(wd_csv):
    df = pd.read_csv(wd_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(df['weight_decay'], df['best_val_acc'], marker='o')
    plt.xscale('log')
    plt.xlabel('Weight Decay')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Weight Decay Search Results')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'weight_decay_search_plot.png'))
    plt.close()
else:
    print('No weight_decay_search.csv found.')

# --- Config Summary ---
config_json = os.path.join(OUTPUT_DIR, 'run_config.json')
if os.path.exists(config_json):
    with open(config_json, 'r') as f:
        config = json.load(f)
    print('Run Config:')
    print(json.dumps(config, indent=2))
else:
    print('No run_config.json found.')

print('Analysis complete. Plots saved in', OUTPUT_DIR)
