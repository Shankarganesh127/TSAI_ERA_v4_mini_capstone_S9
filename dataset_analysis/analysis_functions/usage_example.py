"""
Example usage of the dataset_analyzer module
"""

# Import the module
from dataset_analyzer import *

# Example 1: Analyze CSV dataset
def analyze_csv_example():
    # Load dataset
    df = load_dataset('your_data.csv')

    # Run complete analysis
    report = complete_analysis_pipeline(
        df=df,
        dataset_name="MyDataset",
        target_col="target_column",  # Optional
        generate_plots=True,
        save_outputs=True
    )

    print("Analysis complete!")
    return report

# Example 2: Analyze image dataset
def analyze_images_example():
    # Load image dataset info
    dataset_info = load_image_dataset_info('/path/to/image/dataset')

    # Visualize samples
    visualize_image_samples(dataset_info, samples_per_class=3)

    print("Image analysis complete!")
    return dataset_info

if __name__ == "__main__":
    print("Dataset Analysis Usage Examples")
    print("Uncomment the function calls below to run examples:")
    print("# analyze_csv_example()")
    print("# analyze_images_example()")
