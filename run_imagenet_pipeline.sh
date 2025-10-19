#!/bin/bash
# ImageNet Training Pipeline Launcher
# Provides easy access to different pipeline modes

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_PATH=""
OUTPUT_DIR="./imagenet_pipeline_results"
MODE="full"
EPOCHS=90
BATCH_SIZE=""

print_usage() {
    echo -e "${BLUE}ImageNet Training Pipeline Launcher${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data PATH       Path to ImageNet dataset (required)"
    echo "  -o, --output DIR      Output directory (default: ./imagenet_pipeline_results)"
    echo "  -m, --mode MODE       Pipeline mode: full, quick, test, custom (default: full)"
    echo "  -e, --epochs N        Number of epochs (default: 90)"
    echo "  -b, --batch-size N    Batch size (auto-detect if not specified)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Modes:"
    echo "  full    - Complete 7-step pipeline (recommended for production)"
    echo "  quick   - Fast mode for testing (reduced iterations)"
    echo "  test    - Very fast mode for validation (minimal epochs)"
    echo "  custom  - Skip LR test and WD search (for experienced users)"
    echo ""
    echo "Examples:"
    echo "  $0 --data /datasets/imagenet --mode full"
    echo "  $0 -d /data/imagenet -m quick -e 20 -o ./quick_results"
    echo "  $0 --data ./imagenet --mode test --batch-size 128"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_PATH" ]]; then
    echo -e "${RED}Error: Data path is required${NC}"
    echo -e "${YELLOW}Use --data /path/to/imagenet${NC}"
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    echo -e "${RED}Error: Data path does not exist: $DATA_PATH${NC}"
    exit 1
fi

# Check for ImageNet structure
if [[ ! -d "$DATA_PATH/train" ]] || [[ ! -d "$DATA_PATH/val" ]]; then
    echo -e "${YELLOW}Warning: ImageNet structure not detected${NC}"
    echo -e "${YELLOW}Expected: $DATA_PATH/train and $DATA_PATH/val${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

case $MODE in
    full)
        echo -e "${GREEN}🚀 Running FULL pipeline (all 7 steps)${NC}"
        echo -e "${BLUE}This will take several hours to complete${NC}"
        ;;
    quick)
        echo -e "${YELLOW}⚡ Running QUICK pipeline (reduced iterations)${NC}"
        EPOCHS=20  # Update epochs variable for display
        ;;
    test)
        echo -e "${YELLOW}🧪 Running TEST pipeline (minimal validation)${NC}"
        EPOCHS=5  # Update epochs variable for display
        ;;
    custom)
        echo -e "${BLUE}🎛️  Running CUSTOM pipeline (skip LR test & WD search)${NC}"
        ;;
    *)
        echo -e "${RED}Error: Unknown mode: $MODE${NC}"
        echo -e "${YELLOW}Available modes: full, quick, test, custom${NC}"
        exit 1
        ;;
esac

# Build command based on mode
CMD="uv run python imagenet_training_pipeline.py --data \"$DATA_PATH\" --output \"$OUTPUT_DIR\" --epochs $EPOCHS"

# Add batch size if specified
if [[ -n "$BATCH_SIZE" ]]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

# Add mode-specific flags
case $MODE in
    quick)
        CMD="$CMD --quick-mode"
        ;;
    test)
        CMD="$CMD --quick-mode"
        ;;
    custom)
        CMD="$CMD --skip-lr-test --skip-wd-search"
        ;;
esac

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Data Path: $DATA_PATH"
echo -e "  📁 Output: $OUTPUT_DIR"
echo -e "  🔄 Mode: $MODE"
echo -e "  📚 Epochs: $EPOCHS"
if [[ -n "$BATCH_SIZE" ]]; then
    echo -e "  📦 Batch Size: $BATCH_SIZE"
else
    echo -e "  📦 Batch Size: Auto-detect"
fi
echo ""

# Confirm execution
read -p "Start training pipeline? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo -e "${YELLOW}Training cancelled${NC}"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log start time
echo -e "${GREEN}🎯 Starting ImageNet Training Pipeline at $(date)${NC}"
echo "Command: $CMD"
echo ""

# Execute the pipeline
eval $CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}🎉 Pipeline completed successfully!${NC}"
    echo -e "${BLUE}📊 Results saved to: $OUTPUT_DIR${NC}"
    echo ""
    
    # Show summary if results exist
    if [[ -f "$OUTPUT_DIR/final_results.json" ]]; then
        echo -e "${BLUE}📈 Final Results Summary:${NC}"
        uv run python -c "
import json
try:
    with open('$OUTPUT_DIR/final_results.json', 'r') as f:
        results = json.load(f)
    print(f'  🎯 Best Val Accuracy: {results.get(\"best_val_acc\", \"N/A\"):.2f}%')
    print(f'  📦 Optimal Batch Size: {results.get(\"batch_size\", \"N/A\")}')
    print(f'  ⚖️  Best Weight Decay: {results.get(\"weight_decay\", \"N/A\"):.2e}')
    print(f'  📏 LR Range: {results.get(\"lr_config\", {}).get(\"min_lr\", \"N/A\"):.2e} → {results.get(\"lr_config\", {}).get(\"max_lr\", \"N/A\"):.2e}')
except:
    print('  Results file not found or corrupted')
"
    fi
    
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  1. Review training curves: $OUTPUT_DIR/training_results.png"
    echo -e "  2. Check detailed logs: $OUTPUT_DIR/pipeline.log"
    echo -e "  3. Use best model: $OUTPUT_DIR/best_model.pth"
else
    echo ""
    echo -e "${RED}❌ Pipeline failed with exit code $?${NC}"
    echo -e "${YELLOW}Check the logs for details: $OUTPUT_DIR/pipeline.log${NC}"
    exit 1
fi