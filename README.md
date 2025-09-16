# Flow Cytometry Data Denoising Pipeline

This repository contains a comprehensive Python pipeline for processing flow cytometry data, specifically designed to identify and remove noise from FCS (Flow Cytometry Standard) files.

## Overview

The pipeline processes two FCS files:
- `full_measurement.fcs`: Contains the complete measurement data (41,350 events)
- `only_noise.fcs`: Contains noise-only data (32,875 events)

The goal is to merge the data while retaining indices, identify noise patterns from the noise-only file within the combined dataset, and implement denoising techniques specifically for data with FL1 > 2×10⁴.

## Features

- **FCS File Parsing**: Custom implementation to read FCS 3.1 format files
- **Data Merging**: Combines datasets while preserving original indices
- **Threshold Filtering**: Applies FL1 > 2×10⁴ threshold as specified
- **Multiple Noise Detection Methods**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - DBSCAN Clustering
  - Ensemble Method (majority vote)
- **Comprehensive Analysis**: Statistical analysis, visualizations, and performance metrics
- **Denoising Implementation**: Removes detected noise and measures accuracy

## Requirements

- Python 3.12+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

Install the required packages using apt (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y python3-numpy python3-pandas python3-matplotlib python3-seaborn python3-sklearn
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python3 flow_cytometry_pipeline.py
```

### Individual Components

Parse a single FCS file:

```bash
python3 fcs_parser.py full_measurement.fcs
```

### Pipeline Components

1. **Data Loading**: Loads and combines both FCS files with source labels
2. **FL1 Filtering**: Applies the FL1 > 2×10⁴ threshold
3. **Data Exploration**: Generates distributions and correlation analysis
4. **Noise Detection**: Applies multiple algorithms to identify noise patterns
5. **Visualization**: Creates comprehensive plots for analysis
6. **Denoising**: Removes detected noise and measures performance
7. **Reporting**: Generates detailed performance metrics

## Results

After running the pipeline with the provided data:

### Data Processing
- **Original Events**: 74,225 (41,350 full + 32,875 noise)
- **After FL1 > 2×10⁴ filtering**: 6,421 events (91.3% removed)
- **Filtered Data Composition**: 99.4% noise, 0.6% signal

### Noise Detection Performance
- **Isolation Forest**: 39.7% accuracy
- **Local Outlier Factor**: 39.7% accuracy  
- **DBSCAN**: 18.2% accuracy
- **Ensemble Method**: 30.0% accuracy

### Denoising Results
- **Precision**: 98.7% (very low false positive rate)
- **Recall**: 29.9% (captures some but not all noise)
- **F1-Score**: 45.9%
- **Data Reduction**: 30.2% of filtered data removed as noise

### Key Insights

1. **Threshold Effect**: The FL1 > 2×10⁴ threshold primarily selects events from the noise file, indicating that noise events have higher FL1 values than signal events.

2. **Detection Challenge**: The high noise proportion (99.4%) after filtering makes traditional anomaly detection challenging, as algorithms are designed for scenarios with lower contamination rates.

3. **High Precision**: The ensemble method achieves very high precision (98.7%), meaning when it identifies something as noise, it's almost always correct.

4. **Conservative Approach**: The low recall (29.9%) indicates the method is conservative, missing some noise but avoiding false positives.

## Output Files

The pipeline generates several output files:

- `denoised_data.csv`: Cleaned dataset with noise removed
- `parameter_distributions.png`: Original parameter distributions by source
- `correlation_matrix.png`: Parameter correlation heatmap
- `noise_detection_results.png`: PCA visualization of detection results
- `denoising_comparison.png`: Before/after denoising comparison

## Implementation Details

### FCS Parser
- Supports FCS 3.1 format
- Handles both integer and float data types
- Automatically detects byte order (little/big endian)
- Extracts parameter names from metadata

### Noise Detection Algorithms
1. **Isolation Forest**: Tree-based anomaly detection
2. **Local Outlier Factor**: Density-based outlier detection
3. **DBSCAN**: Clustering-based approach
4. **Ensemble**: Majority vote across all methods

### Evaluation Metrics
- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall

## Recommendations

1. **Method Selection**: Local Outlier Factor and Isolation Forest performed best for this dataset
2. **Parameter Tuning**: Consider adjusting contamination rates based on expected noise levels
3. **Feature Engineering**: Additional features derived from temporal or spatial relationships might improve detection
4. **Validation**: Use additional labeled datasets to validate the approach

## Limitations

1. **High Noise Ratio**: The 99.4% noise ratio after filtering is challenging for standard anomaly detection
2. **Limited Ground Truth**: Performance depends on the assumption that the source labels accurately represent noise vs. signal
3. **Parameter Sensitivity**: Algorithm performance may vary with different parameter settings

## Future Improvements

1. **Advanced Features**: Incorporate temporal patterns and multi-parameter relationships
2. **Deep Learning**: Explore neural network-based approaches for complex pattern recognition
3. **Active Learning**: Implement user feedback to improve detection accuracy
4. **Real-time Processing**: Optimize for streaming data processing

## License

This project is provided as-is for research and educational purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the pipeline.