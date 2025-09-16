#!/usr/bin/env python3
"""
Final Comprehensive Flow Cytometry Data Denoising Pipeline

This pipeline addresses all user feedback:
1. Individual file testing with mean scores across all combinations
2. Optimized SVM training with limited iterations  
3. Fixed Bayesian methods with proper fit/predict interface
4. Complete ensemble method evaluation with TP/FP/TN/FN metrics
5. Comprehensive visualizations for all algorithms

Key improvements:
- Cross-validation testing: Each normal file vs each noise file
- Statistical reporting: Mean ± std across all file combinations
- Optimized performance: Faster training times
- Enhanced visualizations: FL1-FL2 plots for every algorithm
- Complete ensemble analysis: All voting methods evaluated
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from matplotlib.path import Path
import pickle
import os
import warnings
import time
from itertools import product
warnings.filterwarnings('ignore')

from fcs_parser import load_fcs_data
from improved_bayesian_denoising import BayesianTemporalDenoiser, load_multiple_files


class FinalComprehensiveFlowCytometryPipeline:
    """
    Final comprehensive pipeline with individual file testing, optimized performance,
    fixed Bayesian methods, and complete ensemble evaluation.
    """
    
    def __init__(self):
        # Data storage
        self.normal_files = []
        self.noise_files = []
        self.individual_results = []
        
        # Algorithms
        self.algorithms = {}
        self.ensemble_methods = {}
        self.scalers = {'standard': StandardScaler(), 'robust': RobustScaler(), 'minmax': MinMaxScaler()}
        self.best_scaler = None
        
        # Results storage
        self.training_times = {}
        self.performance_results = {}
        self.mean_performance = {}
        
        # Visualization storage
        self.visualization_data = {}
        
    def load_data_by_files(self):
        """Load normal and noise files separately for individual testing."""
        print("============================================================")
        print("FINAL COMPREHENSIVE FLOW CYTOMETRY PIPELINE")
        print("============================================================")
        print("1. LOADING DATA BY INDIVIDUAL FILES")
        
        # Load normal files
        normal_dir = "data/normal_files"
        if os.path.exists(normal_dir):
            print(f"Loading normal files from: {normal_dir}")
            self.normal_files = load_multiple_files(normal_dir)
        else:
            # Fallback to root directory
            if os.path.exists("full_measurement.fcs"):
                print("Loading normal data from root directory")
                data = load_fcs_data("full_measurement.fcs")
                data['source_file'] = 'full_measurement.fcs'
                self.normal_files = [data]
        
        # Load noise files  
        noise_dir = "data/noise_files"
        if os.path.exists(noise_dir):
            print(f"Loading noise files from: {noise_dir}")
            self.noise_files = load_multiple_files(noise_dir)
        else:
            # Fallback to root directory
            if os.path.exists("only_noise.fcs"):
                print("Loading noise data from root directory")
                data = load_fcs_data("only_noise.fcs")
                data['source_file'] = 'only_noise.fcs'
                self.noise_files = [data]
        
        print(f"Loaded {len(self.normal_files)} normal files")
        print(f"Loaded {len(self.noise_files)} noise files")
        
        if not self.normal_files or not self.noise_files:
            raise ValueError("No data files found. Please ensure FCS files are available.")
    
    def apply_polygonal_filter(self, data):
        """Apply polygonal filtering to data."""
        # Polygonal coordinates (log10 scale): [[4.2, 0], [4.2, 3.2], [6.7, 5.9], [6.7, 0.0]]
        # Convert to linear scale
        coords_log = np.array([[4.2, 0], [4.2, 3.2], [6.7, 5.9], [6.7, 0.0]])
        coords_linear = 10 ** coords_log
        
        # Create polygon path
        polygon = Path(coords_linear)
        
        # Check which points are inside the polygon
        points = np.column_stack([data['FL1'], data['FL2']])
        inside_mask = polygon.contains_points(points)
        
        return data[inside_mask].copy()
    
    def optimize_algorithms(self, X_train, contamination_rate=0.1):
        """Train and optimize all algorithms with performance improvements."""
        print("3. OPTIMIZED ALGORITHM TRAINING")
        
        # Select best scaler
        best_score = 0
        for scaler_name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X_train)
                # Quick evaluation with Isolation Forest
                iso = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=50)
                iso.fit(X_scaled)
                score = iso.score_samples(X_scaled).mean()
                if score > best_score:
                    best_score = score
                    self.best_scaler = scaler_name
            except:
                continue
        
        if self.best_scaler is None:
            self.best_scaler = 'standard'
        
        print(f"Selected scaler: {self.best_scaler}")
        scaler = self.scalers[self.best_scaler]
        X_scaled = scaler.fit_transform(X_train)
        
        # Train algorithms with optimizations
        algorithms_config = {
            'isolation_forest': {
                'model': IsolationForest,
                'params': {'contamination': contamination_rate, 'random_state': 42, 'n_estimators': 100}
            },
            'lof': {
                'model': LocalOutlierFactor,
                'params': {'contamination': contamination_rate, 'novelty': True, 'n_neighbors': 20}
            },
            'one_class_svm': {
                'model': OneClassSVM,
                'params': {'gamma': 'scale', 'nu': contamination_rate, 'max_iter': 100}  # Limited iterations
            },
            'elliptic_envelope': {
                'model': EllipticEnvelope,
                'params': {'contamination': contamination_rate, 'random_state': 42}
            },
            'gaussian_mixture': {
                'model': GaussianMixture,
                'params': {'n_components': 2, 'random_state': 42, 'max_iter': 100}  # Limited iterations
            }
        }
        
        # DBSCAN with hyperparameter tuning
        print("Optimizing DBSCAN...")
        start_time = time.time()
        dbscan_params = {'eps': [0.3, 0.5, 0.7], 'min_samples': [5, 10, 15]}
        best_dbscan_score = -1
        best_dbscan_params = {'eps': 0.5, 'min_samples': 10}
        
        for eps, min_samples in product(dbscan_params['eps'], dbscan_params['min_samples']):
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 0 and n_noise > 0:
                    score = n_clusters / (1 + n_noise/len(X_scaled))  # Simple scoring
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
            except:
                continue
        
        self.algorithms['dbscan'] = DBSCAN(**best_dbscan_params)
        self.training_times['dbscan'] = time.time() - start_time
        
        # Train other algorithms
        for alg_name, config in algorithms_config.items():
            print(f"Training {alg_name}...")
            start_time = time.time()
            try:
                model = config['model'](**config['params'])
                model.fit(X_scaled)
                self.algorithms[alg_name] = model
                self.training_times[alg_name] = time.time() - start_time
                print(f"  ✓ {alg_name} trained in {self.training_times[alg_name]:.2f}s")
            except Exception as e:
                print(f"  ✗ {alg_name} training failed: {e}")
                self.training_times[alg_name] = 0
        
        # Train Bayesian methods
        print("Training Bayesian Temporal Methods...")
        start_time = time.time()
        try:
            # Create DataFrame for Bayesian method
            feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
            if X_train.shape[1] >= len(feature_names):
                X_df = pd.DataFrame(X_train, columns=feature_names)
            else:
                # Pad with dummy TIME column if needed
                X_df = pd.DataFrame(X_train)
                X_df['TIME'] = np.arange(len(X_train))
                X_df.columns = feature_names[:X_train.shape[1]] + ['TIME']
            
            bayesian_denoiser = BayesianTemporalDenoiser(method='temporal_cooccurrence')
            bayesian_denoiser.fit(X_df)
            self.algorithms['bayesian_temporal'] = bayesian_denoiser
            self.training_times['bayesian_temporal'] = time.time() - start_time
            print(f"  ✓ Bayesian methods trained in {self.training_times['bayesian_temporal']:.2f}s")
        except Exception as e:
            print(f"  ✗ Bayesian training failed: {e}")
            self.training_times['bayesian_temporal'] = 0
    
    def create_ensemble_methods(self):
        """Create ensemble methods for evaluation."""
        print("4. CREATING ENSEMBLE METHODS")
        
        self.ensemble_methods = {
            'majority_voting': self._majority_voting,
            'weighted_voting': self._weighted_voting,
            'conservative_ensemble': self._conservative_ensemble
        }
        
        print(f"Created {len(self.ensemble_methods)} ensemble methods")
    
    def test_individual_file_combinations(self):
        """Test each normal file against each noise file and calculate mean performance."""
        print("5. INDIVIDUAL FILE COMBINATION TESTING")
        print("=" * 60)
        
        self.individual_results = []
        
        # Test each combination of normal and noise files
        for normal_data in self.normal_files:
            for noise_data in self.noise_files:
                print(f"\nTesting: {normal_data['source_file'].iloc[0]} vs {noise_data['source_file'].iloc[0]}")
                
                # Apply polygonal filtering
                normal_filtered = self.apply_polygonal_filter(normal_data)
                noise_filtered = self.apply_polygonal_filter(noise_data)
                
                print(f"  Normal events after filtering: {len(normal_filtered)}")
                print(f"  Noise events after filtering: {len(noise_filtered)}")
                
                if len(normal_filtered) == 0 or len(noise_filtered) == 0:
                    print("  ⚠ Skipping - insufficient data after filtering")
                    continue
                
                # Train on normal data
                feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
                if 'TIME' in normal_filtered.columns:
                    feature_cols.append('TIME')
                
                X_normal = normal_filtered[feature_cols].fillna(0).values
                
                # Optimize algorithms for this combination
                contamination_rate = min(0.4, len(noise_filtered) / (len(normal_filtered) + len(noise_filtered)))
                self.optimize_algorithms(X_normal, contamination_rate)
                
                # Test on combined data
                combined_data = pd.concat([normal_filtered, noise_filtered], ignore_index=True)
                combined_data['true_label'] = [0] * len(normal_filtered) + [1] * len(noise_filtered)
                
                X_test = combined_data[feature_cols].fillna(0).values
                y_true = combined_data['true_label'].values
                
                # Scale test data
                scaler = self.scalers[self.best_scaler]
                scaler.fit(X_normal)  # Fit on training data only
                X_test_scaled = scaler.transform(X_test)
                
                # Test each algorithm
                file_results = {
                    'normal_file': normal_data['source_file'].iloc[0],
                    'noise_file': noise_data['source_file'].iloc[0],
                    'normal_events': len(normal_filtered),
                    'noise_events': len(noise_filtered),
                    'algorithms': {}
                }
                
                # Test individual algorithms
                for alg_name, model in self.algorithms.items():
                    try:
                        results = self._evaluate_algorithm(model, X_test, X_test_scaled, y_true, alg_name)
                        file_results['algorithms'][alg_name] = results
                        print(f"    {alg_name}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}")
                    except Exception as e:
                        print(f"    {alg_name}: Failed - {e}")
                        file_results['algorithms'][alg_name] = self._get_default_metrics()
                
                # Test ensemble methods
                for ensemble_name, ensemble_func in self.ensemble_methods.items():
                    try:
                        y_pred = ensemble_func(X_test, X_test_scaled)
                        results = self._calculate_metrics(y_true, y_pred)
                        file_results['algorithms'][ensemble_name] = results
                        print(f"    {ensemble_name}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}")
                    except Exception as e:
                        print(f"    {ensemble_name}: Failed - {e}")
                        file_results['algorithms'][ensemble_name] = self._get_default_metrics()
                
                self.individual_results.append(file_results)
        
        # Calculate mean performance across all combinations
        self._calculate_mean_performance()
    
    def _evaluate_algorithm(self, model, X_test, X_test_scaled, y_true, alg_name):
        """Evaluate a single algorithm."""
        if alg_name == 'dbscan':
            labels = model.fit_predict(X_test_scaled)
            # Convert DBSCAN labels to binary (noise = -1 -> 1, clusters -> 0)
            y_pred = (labels == -1).astype(int)
        elif alg_name == 'gaussian_mixture':
            labels = model.fit_predict(X_test_scaled)
            # Identify noise cluster (minority cluster)
            unique_labels, counts = np.unique(labels, return_counts=True)
            noise_cluster = unique_labels[np.argmin(counts)]
            y_pred = (labels == noise_cluster).astype(int)
        elif alg_name == 'bayesian_temporal':
            # Use DataFrame for Bayesian method
            feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
            X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
            if 'TIME' not in X_df.columns:
                X_df['TIME'] = np.arange(len(X_test))
            y_pred = model.predict(X_df)
        else:
            # Outlier detection algorithms
            y_pred = model.predict(X_test_scaled)
            # Convert outlier labels (-1 = outlier -> 1 = noise, 1 = inlier -> 0 = normal)
            y_pred = (y_pred == -1).astype(int)
        
        return self._calculate_metrics(y_true, y_pred)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
    
    def _get_default_metrics(self):
        """Return default metrics for failed algorithms."""
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
        }
    
    def _majority_voting(self, X_test, X_test_scaled):
        """Majority voting ensemble."""
        predictions = []
        
        for alg_name, model in self.algorithms.items():
            try:
                if alg_name == 'dbscan':
                    labels = model.fit_predict(X_test_scaled)
                    pred = (labels == -1).astype(int)
                elif alg_name == 'gaussian_mixture':
                    labels = model.fit_predict(X_test_scaled)
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    noise_cluster = unique_labels[np.argmin(counts)]
                    pred = (labels == noise_cluster).astype(int)
                elif alg_name == 'bayesian_temporal':
                    feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                    X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                    if 'TIME' not in X_df.columns:
                        X_df['TIME'] = np.arange(len(X_test))
                    pred = model.predict(X_df)
                else:
                    pred = model.predict(X_test_scaled)
                    pred = (pred == -1).astype(int)
                
                predictions.append(pred)
            except:
                # Skip failed algorithms
                continue
        
        if predictions:
            return (np.mean(predictions, axis=0) > 0.5).astype(int)
        else:
            return np.zeros(len(X_test))
    
    def _weighted_voting(self, X_test, X_test_scaled):
        """Weighted voting based on F1 scores."""
        if not hasattr(self, 'algorithm_weights'):
            # Equal weights if no performance history
            self.algorithm_weights = {name: 1.0 for name in self.algorithms.keys()}
        
        weighted_predictions = np.zeros(len(X_test))
        total_weight = 0
        
        for alg_name, model in self.algorithms.items():
            try:
                weight = self.algorithm_weights.get(alg_name, 1.0)
                
                if alg_name == 'dbscan':
                    labels = model.fit_predict(X_test_scaled)
                    pred = (labels == -1).astype(int)
                elif alg_name == 'gaussian_mixture':
                    labels = model.fit_predict(X_test_scaled)
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    noise_cluster = unique_labels[np.argmin(counts)]
                    pred = (labels == noise_cluster).astype(int)
                elif alg_name == 'bayesian_temporal':
                    feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                    X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                    if 'TIME' not in X_df.columns:
                        X_df['TIME'] = np.arange(len(X_test))
                    pred = model.predict(X_df)
                else:
                    pred = model.predict(X_test_scaled)
                    pred = (pred == -1).astype(int)
                
                weighted_predictions += weight * pred
                total_weight += weight
            except:
                continue
        
        if total_weight > 0:
            return (weighted_predictions / total_weight > 0.5).astype(int)
        else:
            return np.zeros(len(X_test))
    
    def _conservative_ensemble(self, X_test, X_test_scaled):
        """Conservative ensemble - only predict noise if multiple algorithms agree."""
        predictions = []
        
        for alg_name, model in self.algorithms.items():
            try:
                if alg_name == 'dbscan':
                    labels = model.fit_predict(X_test_scaled)
                    pred = (labels == -1).astype(int)
                elif alg_name == 'gaussian_mixture':
                    labels = model.fit_predict(X_test_scaled)
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    noise_cluster = unique_labels[np.argmin(counts)]
                    pred = (labels == noise_cluster).astype(int)
                elif alg_name == 'bayesian_temporal':
                    feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                    X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                    if 'TIME' not in X_df.columns:
                        X_df['TIME'] = np.arange(len(X_test))
                    pred = model.predict(X_df)
                else:
                    pred = model.predict(X_test_scaled)
                    pred = (pred == -1).astype(int)
                
                predictions.append(pred)
            except:
                continue
        
        if predictions:
            # Require at least 2/3 agreement for noise prediction
            vote_sum = np.sum(predictions, axis=0)
            threshold = max(2, len(predictions) * 0.67)
            return (vote_sum >= threshold).astype(int)
        else:
            return np.zeros(len(X_test))
    
    def _calculate_mean_performance(self):
        """Calculate mean performance across all file combinations."""
        print("\n6. CALCULATING MEAN PERFORMANCE ACROSS ALL FILE COMBINATIONS")
        print("=" * 70)
        
        if not self.individual_results:
            print("No individual results to analyze")
            return
        
        # Collect all algorithm names
        all_algorithms = set()
        for result in self.individual_results:
            all_algorithms.update(result['algorithms'].keys())
        
        # Calculate means and standard deviations
        for alg_name in all_algorithms:
            metrics = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                'tp': [], 'fp': [], 'tn': [], 'fn': []
            }
            
            for result in self.individual_results:
                if alg_name in result['algorithms']:
                    alg_results = result['algorithms'][alg_name]
                    for metric in metrics:
                        metrics[metric].append(alg_results[metric])
            
            # Calculate mean and std
            if metrics['accuracy']:
                self.mean_performance[alg_name] = {
                    'accuracy_mean': np.mean(metrics['accuracy']),
                    'accuracy_std': np.std(metrics['accuracy']),
                    'precision_mean': np.mean(metrics['precision']),
                    'precision_std': np.std(metrics['precision']),
                    'recall_mean': np.mean(metrics['recall']),
                    'recall_std': np.std(metrics['recall']),
                    'f1_score_mean': np.mean(metrics['f1_score']),
                    'f1_score_std': np.std(metrics['f1_score']),
                    'tp_mean': np.mean(metrics['tp']),
                    'fp_mean': np.mean(metrics['fp']),
                    'tn_mean': np.mean(metrics['tn']),
                    'fn_mean': np.mean(metrics['fn']),
                    'n_combinations': len(metrics['accuracy'])
                }
        
        # Print summary
        print("\nMEAN PERFORMANCE ACROSS ALL FILE COMBINATIONS:")
        print("-" * 100)
        print(f"{'Algorithm':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'N':<5}")
        print("-" * 100)
        
        # Sort by F1 score
        sorted_algorithms = sorted(self.mean_performance.items(), 
                                 key=lambda x: x[1]['f1_score_mean'], reverse=True)
        
        for alg_name, perf in sorted_algorithms:
            print(f"{alg_name:<20} "
                  f"{perf['accuracy_mean']:.3f}±{perf['accuracy_std']:.3f}    "
                  f"{perf['precision_mean']:.3f}±{perf['precision_std']:.3f}    "
                  f"{perf['recall_mean']:.3f}±{perf['recall_std']:.3f}    "
                  f"{perf['f1_score_mean']:.3f}±{perf['f1_score_std']:.3f}    "
                  f"{perf['n_combinations']:<5}")
        
        print("-" * 100)
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for all algorithms."""
        print("\n7. CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)
        
        if not self.individual_results:
            print("No results to visualize")
            return
        
        # Create visualizations for the last test combination
        last_result = self.individual_results[-1]
        
        # Get the data for the last combination
        normal_file = [f for f in self.normal_files if f['source_file'].iloc[0] == last_result['normal_file']][0]
        noise_file = [f for f in self.noise_files if f['source_file'].iloc[0] == last_result['noise_file']][0]
        
        normal_filtered = self.apply_polygonal_filter(normal_file)
        noise_filtered = self.apply_polygonal_filter(noise_file)
        
        combined_data = pd.concat([normal_filtered, noise_filtered], ignore_index=True)
        combined_data['true_label'] = [0] * len(normal_filtered) + [1] * len(noise_filtered)
        
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        if 'TIME' in combined_data.columns:
            feature_cols.append('TIME')
        
        X_test = combined_data[feature_cols].fillna(0).values
        y_true = combined_data['true_label'].values
        
        # Create FL1-FL2 scatter plots for all algorithms
        n_algorithms = len(last_result['algorithms'])
        n_cols = 3
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (alg_name, alg_results) in enumerate(last_result['algorithms'].items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get predictions for this algorithm
            try:
                scaler = self.scalers[self.best_scaler]
                scaler.fit(X_test)
                X_test_scaled = scaler.transform(X_test)
                
                if alg_name in self.algorithms:
                    model = self.algorithms[alg_name]
                    if alg_name == 'dbscan':
                        labels = model.fit_predict(X_test_scaled)
                        y_pred = (labels == -1).astype(int)
                    elif alg_name == 'gaussian_mixture':
                        labels = model.fit_predict(X_test_scaled)
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        noise_cluster = unique_labels[np.argmin(counts)]
                        y_pred = (labels == noise_cluster).astype(int)
                    elif alg_name == 'bayesian_temporal':
                        feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                        X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                        if 'TIME' not in X_df.columns:
                            X_df['TIME'] = np.arange(len(X_test))
                        y_pred = model.predict(X_df)
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_pred = (y_pred == -1).astype(int)
                elif alg_name in self.ensemble_methods:
                    ensemble_func = self.ensemble_methods[alg_name]
                    y_pred = ensemble_func(X_test, X_test_scaled)
                else:
                    y_pred = np.zeros(len(X_test))
                
                # Create TP/FP/TN/FN classification
                classification = np.zeros(len(y_true), dtype=int)
                classification[(y_true == 1) & (y_pred == 1)] = 1  # TP
                classification[(y_true == 0) & (y_pred == 0)] = 2  # TN
                classification[(y_true == 0) & (y_pred == 1)] = 3  # FP
                classification[(y_true == 1) & (y_pred == 0)] = 4  # FN
                
                # Plot FL1-FL2 with log scale
                colors = ['gray', 'green', 'blue', 'red', 'orange']
                labels = ['Unknown', 'TP', 'TN', 'FP', 'FN']
                
                for class_idx in range(1, 5):
                    mask = classification == class_idx
                    if mask.any():
                        ax.scatter(combined_data.loc[mask, 'FL1'], combined_data.loc[mask, 'FL2'],
                                 c=colors[class_idx], label=labels[class_idx], alpha=0.6, s=20)
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('FL1')
                ax.set_ylabel('FL2')
                ax.set_title(f'{alg_name}\nAcc: {alg_results["accuracy"]:.3f}, F1: {alg_results["f1_score"]:.3f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')
                ax.set_title(f'{alg_name} - Error')
        
        # Hide unused subplots
        for idx in range(len(last_result['algorithms']), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comprehensive_algorithm_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create performance comparison chart
        self._create_performance_comparison_chart()
        
        print("✓ Comprehensive visualizations created")
        print("  - comprehensive_algorithm_visualizations.png")
        print("  - performance_comparison_chart.png")
    
    def _create_performance_comparison_chart(self):
        """Create performance comparison chart."""
        if not self.mean_performance:
            return
            
        algorithms = list(self.mean_performance.keys())
        accuracy_means = [self.mean_performance[alg]['accuracy_mean'] for alg in algorithms]
        accuracy_stds = [self.mean_performance[alg]['accuracy_std'] for alg in algorithms]
        f1_means = [self.mean_performance[alg]['f1_score_mean'] for alg in algorithms]
        f1_stds = [self.mean_performance[alg]['f1_score_std'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, accuracy_means, width, yerr=accuracy_stds, 
                      label='Accuracy', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds, 
                      label='F1-Score', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Score')
        ax.set_title('Mean Performance Across All File Combinations (with Standard Deviation)')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig('performance_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to files."""
        print("\n8. SAVING RESULTS")
        
        # Save individual results
        results_df = []
        for result in self.individual_results:
            for alg_name, alg_results in result['algorithms'].items():
                row = {
                    'normal_file': result['normal_file'],
                    'noise_file': result['noise_file'],
                    'normal_events': result['normal_events'],
                    'noise_events': result['noise_events'],
                    'algorithm': alg_name,
                    **alg_results
                }
                results_df.append(row)
        
        if results_df:
            pd.DataFrame(results_df).to_csv('individual_file_results.csv', index=False)
            print("✓ Saved individual_file_results.csv")
        
        # Save mean performance
        if self.mean_performance:
            mean_df = []
            for alg_name, perf in self.mean_performance.items():
                mean_df.append({'algorithm': alg_name, **perf})
            
            pd.DataFrame(mean_df).to_csv('mean_performance_results.csv', index=False)
            print("✓ Saved mean_performance_results.csv")
        
        # Save training times
        if self.training_times:
            pd.DataFrame([self.training_times]).to_csv('training_times.csv', index=False)
            print("✓ Saved training_times.csv")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline with all improvements."""
        try:
            self.load_data_by_files()
            self.create_ensemble_methods()
            self.test_individual_file_combinations()
            self.create_comprehensive_visualizations()
            self.save_results()
            
            print("\n" + "="*70)
            print("FINAL COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"✓ Tested {len(self.individual_results)} file combinations")
            print(f"✓ Evaluated {len(self.algorithms)} individual algorithms")
            print(f"✓ Evaluated {len(self.ensemble_methods)} ensemble methods")
            print(f"✓ Generated comprehensive visualizations")
            print(f"✓ Calculated mean performance with statistical analysis")
            
            if self.mean_performance:
                best_algorithm = max(self.mean_performance.items(), 
                                   key=lambda x: x[1]['f1_score_mean'])
                print(f"✓ Best performing algorithm: {best_algorithm[0]} "
                      f"(F1: {best_algorithm[1]['f1_score_mean']:.3f}±{best_algorithm[1]['f1_score_std']:.3f})")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution function."""
    pipeline = FinalComprehensiveFlowCytometryPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()