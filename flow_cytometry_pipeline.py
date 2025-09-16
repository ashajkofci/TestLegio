#!/usr/bin/env python3
"""
Flow Cytometry Data Denoising Pipeline

This pipeline processes two FCS files:
1. full_measurement.fcs - Contains the full measurement data
2. only_noise.fcs - Contains only noise data

The goal is to merge the data while retaining indices, identify noise patterns,
and implement denoising techniques for data with FL1 > 2×10⁴.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

from fcs_parser import load_fcs_data


class FlowCytometryPipeline:
    """Main pipeline for flow cytometry data processing and denoising."""
    
    def __init__(self):
        self.full_data = None
        self.noise_data = None
        self.combined_data = None
        self.filtered_data = None
        self.scaler = StandardScaler()
        self.fl1_threshold = 2e4  # 2×10⁴
        
    def load_data(self, full_measurement_path: str, noise_path: str):
        """Load both FCS files and prepare data with source labels."""
        print("Loading FCS files...")
        
        # Load the data
        self.full_data = load_fcs_data(full_measurement_path)
        self.noise_data = load_fcs_data(noise_path)
        
        print(f"Full measurement data: {self.full_data.shape}")
        print(f"Noise data: {self.noise_data.shape}")
        
        # Add source labels and original indices
        self.full_data['source'] = 'full_measurement'
        self.full_data['original_index'] = range(len(self.full_data))
        
        self.noise_data['source'] = 'noise_only'
        self.noise_data['original_index'] = range(len(self.noise_data))
        
        # Combine datasets while retaining indices
        self.combined_data = pd.concat([self.full_data, self.noise_data], 
                                     ignore_index=True, sort=False)
        
        print(f"Combined data: {self.combined_data.shape}")
        
    def apply_fl1_threshold(self):
        """Apply FL1 > 2×10⁴ threshold filter."""
        print(f"\nApplying FL1 > {self.fl1_threshold:.0e} threshold...")
        
        before_count = len(self.combined_data)
        self.filtered_data = self.combined_data[self.combined_data['FL1'] > self.fl1_threshold].copy()
        after_count = len(self.filtered_data)
        
        print(f"Events before filtering: {before_count}")
        print(f"Events after filtering: {after_count}")
        print(f"Removed {before_count - after_count} events ({100 * (before_count - after_count) / before_count:.1f}%)")
        
        # Show distribution by source
        source_counts = self.filtered_data['source'].value_counts()
        print(f"\nFiltered data by source:")
        for source, count in source_counts.items():
            percentage = 100 * count / len(self.filtered_data)
            print(f"  {source}: {count} events ({percentage:.1f}%)")
    
    def explore_data(self):
        """Explore the data characteristics and distributions."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic statistics
        numeric_cols = ['TIME', 'SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        
        print("\nFiltered data summary statistics:")
        print(self.filtered_data[numeric_cols].describe())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Flow Cytometry Parameters Distribution (FL1 > 2e4)', fontsize=16)
        
        for i, col in enumerate(numeric_cols):
            row = i // 3
            col_idx = i % 3
            ax = axes[row, col_idx]
            
            # Plot distributions by source
            for source in self.filtered_data['source'].unique():
                data_subset = self.filtered_data[self.filtered_data['source'] == source]
                ax.hist(data_subset[col], alpha=0.6, label=source, bins=50)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col} Distribution')
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation analysis
        print("\nCorrelation matrix:")
        corr_matrix = self.filtered_data[numeric_cols].corr()
        print(corr_matrix)
        
        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter Correlation Matrix (FL1 > 2e4)')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_noise_patterns(self):
        """Detect noise patterns using multiple approaches."""
        print("\n" + "="*50)
        print("NOISE PATTERN DETECTION")
        print("="*50)
        
        # Prepare features for analysis
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        X = self.filtered_data[feature_cols].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # True labels (0 = full_measurement, 1 = noise_only)
        y_true = (self.filtered_data['source'] == 'noise_only').astype(int)
        
        print(f"True noise samples: {y_true.sum()} / {len(y_true)} ({100 * y_true.mean():.1f}%)")
        
        # Adjust contamination rate (cap at 0.4 for these algorithms)
        contamination_rate = min(y_true.mean(), 0.4)
        print(f"Adjusted contamination rate for algorithms: {contamination_rate:.3f}")
        
        # Method 1: Isolation Forest
        print("\n1. Isolation Forest Analysis:")
        iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
        y_pred_iso = iso_forest.fit_predict(X_scaled)
        y_pred_iso_binary = (y_pred_iso == -1).astype(int)  # -1 indicates outlier/noise
        
        iso_accuracy = accuracy_score(y_true, y_pred_iso_binary)
        print(f"   Accuracy: {iso_accuracy:.3f}")
        print(f"   Detected anomalies: {y_pred_iso_binary.sum()} / {len(y_pred_iso_binary)}")
        
        # Method 2: Local Outlier Factor
        print("\n2. Local Outlier Factor Analysis:")
        lof = LocalOutlierFactor(contamination=contamination_rate)
        y_pred_lof = lof.fit_predict(X_scaled)
        y_pred_lof_binary = (y_pred_lof == -1).astype(int)
        
        lof_accuracy = accuracy_score(y_true, y_pred_lof_binary)
        print(f"   Accuracy: {lof_accuracy:.3f}")
        print(f"   Detected anomalies: {y_pred_lof_binary.sum()} / {len(y_pred_lof_binary)}")
        
        # Method 3: DBSCAN Clustering
        print("\n3. DBSCAN Clustering Analysis:")
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Identify noise cluster (largest cluster is likely normal data)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            main_cluster = unique_labels[np.argmax(counts)]
            y_pred_dbscan = (cluster_labels != main_cluster).astype(int)
        else:
            y_pred_dbscan = np.ones(len(cluster_labels))
        
        dbscan_accuracy = accuracy_score(y_true, y_pred_dbscan)
        print(f"   Accuracy: {dbscan_accuracy:.3f}")
        print(f"   Detected anomalies: {y_pred_dbscan.sum()} / {len(y_pred_dbscan)}")
        print(f"   Number of clusters: {len(unique_labels) if len(unique_labels) > 0 else 0}")
        print(f"   Noise points (DBSCAN): {(cluster_labels == -1).sum()}")
        
        # Store predictions for later use
        self.filtered_data['iso_forest_outlier'] = y_pred_iso_binary
        self.filtered_data['lof_outlier'] = y_pred_lof_binary
        self.filtered_data['dbscan_outlier'] = y_pred_dbscan
        
        # Create ensemble prediction
        ensemble_score = (y_pred_iso_binary + y_pred_lof_binary + y_pred_dbscan) / 3
        ensemble_pred = (ensemble_score > 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        
        self.filtered_data['ensemble_outlier'] = ensemble_pred
        
        print(f"\n4. Ensemble Method (majority vote):")
        print(f"   Accuracy: {ensemble_accuracy:.3f}")
        print(f"   Detected anomalies: {ensemble_pred.sum()} / {len(ensemble_pred)}")
        
        return {
            'isolation_forest': iso_accuracy,
            'local_outlier_factor': lof_accuracy,
            'dbscan': dbscan_accuracy,
            'ensemble': ensemble_accuracy
        }
    
    def visualize_noise_detection(self):
        """Visualize noise detection results."""
        print("\nCreating noise detection visualizations...")
        
        # PCA for 2D visualization
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        X = self.filtered_data[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Noise Detection Results (PCA Visualization)', fontsize=16)
        
        # True labels
        axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=self.filtered_data['source'].map({'full_measurement': 0, 'noise_only': 1}),
                          cmap='coolwarm', alpha=0.6)
        axes[0, 0].set_title('True Labels')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Isolation Forest
        axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=self.filtered_data['iso_forest_outlier'],
                          cmap='coolwarm', alpha=0.6)
        axes[0, 1].set_title('Isolation Forest')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Local Outlier Factor
        axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=self.filtered_data['lof_outlier'],
                          cmap='coolwarm', alpha=0.6)
        axes[0, 2].set_title('Local Outlier Factor')
        axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # DBSCAN
        axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=self.filtered_data['dbscan_outlier'],
                          cmap='coolwarm', alpha=0.6)
        axes[1, 0].set_title('DBSCAN')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Ensemble
        axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=self.filtered_data['ensemble_outlier'],
                          cmap='coolwarm', alpha=0.6)
        axes[1, 1].set_title('Ensemble Method')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # FL1 vs FL2 scatter plot with true labels
        axes[1, 2].scatter(self.filtered_data['FL1'], self.filtered_data['FL2'],
                          c=self.filtered_data['source'].map({'full_measurement': 0, 'noise_only': 1}),
                          cmap='coolwarm', alpha=0.6)
        axes[1, 2].set_title('FL1 vs FL2 (True Labels)')
        axes[1, 2].set_xlabel('FL1')
        axes[1, 2].set_ylabel('FL2')
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('noise_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def implement_denoising(self):
        """Implement denoising techniques and measure accuracy."""
        print("\n" + "="*50)
        print("DENOISING IMPLEMENTATION")
        print("="*50)
        
        # Use the best performing method for denoising
        best_method = 'ensemble_outlier'  # Can be adjusted based on results
        
        # Create denoised dataset by removing detected noise
        noise_mask = self.filtered_data[best_method] == 1
        denoised_data = self.filtered_data[~noise_mask].copy()
        
        print(f"Original filtered data: {len(self.filtered_data)} events")
        print(f"Detected noise events: {noise_mask.sum()} events")
        print(f"Denoised data: {len(denoised_data)} events")
        print(f"Removed {100 * noise_mask.sum() / len(self.filtered_data):.1f}% of data as noise")
        
        # Analyze denoising performance
        true_noise_mask = self.filtered_data['source'] == 'noise_only'
        
        # Calculate metrics
        true_positives = (noise_mask & true_noise_mask).sum()
        false_positives = (noise_mask & ~true_noise_mask).sum()
        true_negatives = (~noise_mask & ~true_noise_mask).sum()
        false_negatives = (~noise_mask & true_noise_mask).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nDenoising Performance Metrics:")
        print(f"  True Positives (correctly identified noise): {true_positives}")
        print(f"  False Positives (wrongly identified as noise): {false_positives}")
        print(f"  True Negatives (correctly kept as signal): {true_negatives}")
        print(f"  False Negatives (missed noise): {false_negatives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
        
        # Compare distributions before and after denoising
        self.compare_distributions(denoised_data)
        
        return denoised_data, {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def compare_distributions(self, denoised_data):
        """Compare parameter distributions before and after denoising."""
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Distributions: Before vs After Denoising', fontsize=16)
        
        for i, col in enumerate(feature_cols):
            row = i // 3
            col_idx = i % 3
            ax = axes[row, col_idx]
            
            # Plot original filtered data
            ax.hist(self.filtered_data[col], alpha=0.6, label='Before Denoising', 
                   bins=50, color='red', density=True)
            
            # Plot denoised data
            ax.hist(denoised_data[col], alpha=0.6, label='After Denoising', 
                   bins=50, color='blue', density=True)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'{col} Distribution')
            ax.legend()
            ax.set_yscale('log')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('denoising_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, detection_accuracies, denoising_metrics):
        """Generate a comprehensive report of the pipeline results."""
        print("\n" + "="*70)
        print("FLOW CYTOMETRY DENOISING PIPELINE - FINAL REPORT")
        print("="*70)
        
        print(f"\n1. DATA LOADING AND PREPROCESSING:")
        print(f"   - Full measurement file: {len(self.full_data)} events")
        print(f"   - Noise-only file: {len(self.noise_data)} events")
        print(f"   - Combined dataset: {len(self.combined_data)} events")
        print(f"   - After FL1 > {self.fl1_threshold:.0e} filtering: {len(self.filtered_data)} events")
        
        print(f"\n2. NOISE DETECTION ACCURACY:")
        for method, accuracy in detection_accuracies.items():
            print(f"   - {method.replace('_', ' ').title()}: {accuracy:.3f}")
        
        print(f"\n3. DENOISING PERFORMANCE:")
        for metric, value in denoising_metrics.items():
            if isinstance(value, float):
                print(f"   - {metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"   - {metric.replace('_', ' ').title()}: {value}")
        
        print(f"\n4. RECOMMENDATIONS:")
        best_method = max(detection_accuracies.items(), key=lambda x: x[1])
        print(f"   - Best performing detection method: {best_method[0]} (accuracy: {best_method[1]:.3f})")
        
        if denoising_metrics['precision'] > 0.8:
            print(f"   - High precision ({denoising_metrics['precision']:.3f}): Low false positive rate")
        elif denoising_metrics['precision'] < 0.6:
            print(f"   - Low precision ({denoising_metrics['precision']:.3f}): Consider tuning parameters")
        
        if denoising_metrics['recall'] > 0.8:
            print(f"   - High recall ({denoising_metrics['recall']:.3f}): Successfully captures most noise")
        elif denoising_metrics['recall'] < 0.6:
            print(f"   - Low recall ({denoising_metrics['recall']:.3f}): Missing significant noise")
        
        print(f"\n5. OUTPUT FILES GENERATED:")
        print(f"   - parameter_distributions.png: Original parameter distributions")
        print(f"   - correlation_matrix.png: Parameter correlation analysis")
        print(f"   - noise_detection_results.png: Noise detection visualization")
        print(f"   - denoising_comparison.png: Before/after denoising comparison")
        
        print(f"\n" + "="*70)


def main():
    """Main pipeline execution."""
    # Initialize pipeline
    pipeline = FlowCytometryPipeline()
    
    # Load data
    pipeline.load_data('full_measurement.fcs', 'only_noise.fcs')
    
    # Apply FL1 threshold
    pipeline.apply_fl1_threshold()
    
    # Explore data characteristics
    pipeline.explore_data()
    
    # Detect noise patterns
    detection_accuracies = pipeline.detect_noise_patterns()
    
    # Visualize detection results
    pipeline.visualize_noise_detection()
    
    # Implement denoising
    denoised_data, denoising_metrics = pipeline.implement_denoising()
    
    # Generate final report
    pipeline.generate_report(detection_accuracies, denoising_metrics)
    
    # Save denoised data
    denoised_data.to_csv('denoised_data.csv', index=False)
    print(f"\nDenoised data saved to 'denoised_data.csv'")
    
    return pipeline, denoised_data


if __name__ == "__main__":
    pipeline, denoised_data = main()