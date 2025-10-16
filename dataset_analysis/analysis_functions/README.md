# Dataset Analysis Functions

This folder contains reusable functions for universal dataset analysis.

## Files

- `dataset_analyzer.py` - Main module with all analysis functions
- `usage_example.py` - Example usage patterns
- `README.md` - This documentation

## Quick Start

```python
from dataset_analyzer import *

# Load any dataset
df = load_dataset('data.csv')  # Supports CSV, Excel, JSON, Parquet

# Run complete analysis
report = complete_analysis_pipeline(
    df=df,
    dataset_name="MyData",
    target_col="target",  # Optional
    generate_plots=True,
    save_outputs=True
)
```

## Functions Available

### Data Loading
- `load_dataset()` - Universal data loader
- `load_image_dataset_info()` - Image dataset analysis

### Exploration
- `explore_dataset_basic()` - Basic dataset info
- `explore_numerical_features()` - Numerical analysis
- `explore_categorical_features()` - Categorical analysis

### Visualization
- `plot_missing_values()` - Missing data visualization
- `plot_numerical_distributions()` - Distribution plots
- `plot_correlation_matrix()` - Correlation heatmap
- `plot_categorical_distributions()` - Category bar plots
- `plot_feature_relationships()` - Pairwise relationships
- `plot_feature_importance_simple()` - Feature importance
- `plot_outliers_boxplot()` - Outlier visualization
- `visualize_image_samples()` - Image dataset samples

### Analysis & Reporting
- `generate_dataset_report()` - Comprehensive report
- `assess_data_quality()` - Quality scoring
- `generate_recommendations()` - Actionable insights
- `complete_analysis_pipeline()` - Run everything

## Features

✅ **Universal**: Works with any dataset format
✅ **Comprehensive**: 15+ analysis functions
✅ **Visual**: Automatic plot generation
✅ **Intelligent**: Quality assessment and recommendations
✅ **Exportable**: Save reports and plots
✅ **Modular**: Use individual functions or complete pipeline

Created by Universal Dataset Explorer notebook 🚀
