#!/usr/bin/env python3
"""
Example usage of the Flow Cytometry Denoising Pipeline

This script demonstrates how to use the pipeline components individually
and provides examples of different analysis approaches.
"""

from fcs_parser import load_fcs_data
from flow_cytometry_pipeline import FlowCytometryPipeline
import pandas as pd
import numpy as np


def basic_usage_example():
    """Demonstrate basic usage of the FCS parser."""
    print("="*60)
    print("BASIC FCS FILE LOADING EXAMPLE")
    print("="*60)
    
    # Load individual FCS files
    print("Loading full measurement data...")
    full_data = load_fcs_data('full_measurement.fcs')
    print(f"Loaded {len(full_data)} events with {len(full_data.columns)} parameters")
    print(f"Parameters: {list(full_data.columns)}")
    
    print("\nLoading noise data...")
    noise_data = load_fcs_data('only_noise.fcs')
    print(f"Loaded {len(noise_data)} events with {len(noise_data.columns)} parameters")
    
    # Basic statistics
    print(f"\nFL1 statistics for full measurement:")
    print(f"  Mean: {full_data['FL1'].mean():.1f}")
    print(f"  Median: {full_data['FL1'].median():.1f}")
    print(f"  Events > 2e4: {(full_data['FL1'] > 2e4).sum()}")
    
    print(f"\nFL1 statistics for noise data:")
    print(f"  Mean: {noise_data['FL1'].mean():.1f}")
    print(f"  Median: {noise_data['FL1'].median():.1f}")
    print(f"  Events > 2e4: {(noise_data['FL1'] > 2e4).sum()}")


def pipeline_example():
    """Demonstrate the complete pipeline."""
    print("\n" + "="*60)
    print("COMPLETE PIPELINE EXAMPLE")
    print("="*60)
    
    # Initialize and run pipeline
    pipeline = FlowCytometryPipeline()
    
    # Load data
    pipeline.load_data('full_measurement.fcs', 'only_noise.fcs')
    
    # Apply filtering
    pipeline.apply_fl1_threshold()
    
    # Run noise detection (without full exploration/visualization)
    print("\nRunning noise detection...")
    detection_results = pipeline.detect_noise_patterns()
    
    # Show results
    print("\nDetection Accuracy Summary:")
    for method, accuracy in detection_results.items():
        print(f"  {method.replace('_', ' ').title()}: {accuracy:.3f}")
    
    return pipeline


def custom_analysis_example():
    """Demonstrate custom analysis approaches."""
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    # Load data manually
    full_data = load_fcs_data('full_measurement.fcs')
    noise_data = load_fcs_data('only_noise.fcs')
    
    # Add labels
    full_data['label'] = 'signal'
    noise_data['label'] = 'noise'
    
    # Combine
    combined = pd.concat([full_data, noise_data], ignore_index=True)
    
    # Apply custom filtering
    fl1_threshold = 1e4  # Different threshold
    filtered = combined[combined['FL1'] > fl1_threshold]
    
    print(f"Using FL1 > {fl1_threshold:.0e} threshold:")
    print(f"  Total events: {len(filtered)}")
    
    # Analyze by label
    label_counts = filtered['label'].value_counts()
    print(f"  Signal events: {label_counts.get('signal', 0)}")
    print(f"  Noise events: {label_counts.get('noise', 0)}")
    
    # Parameter comparison
    print(f"\nParameter comparison (FL1 > {fl1_threshold:.0e}):")
    for param in ['SSC', 'FL1', 'FL2', 'FSC']:
        signal_mean = filtered[filtered['label'] == 'signal'][param].mean()
        noise_mean = filtered[filtered['label'] == 'noise'][param].mean()
        print(f"  {param}: Signal={signal_mean:.1f}, Noise={noise_mean:.1f}")


def main():
    """Run all examples."""
    print("Flow Cytometry Pipeline Examples")
    print("================================")
    
    # Basic usage
    basic_usage_example()
    
    # Pipeline example
    pipeline = pipeline_example()
    
    # Custom analysis
    custom_analysis_example()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nFor the complete analysis with visualizations, run:")
    print("  python3 flow_cytometry_pipeline.py")


if __name__ == "__main__":
    main()