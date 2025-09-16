#!/usr/bin/env python3
"""
Optimized Final Flow Cytometry Data Denoising Pipeline

Fixed issues:
1. Train ONCE on all combined normal data (no retraining per test)
2. Test on individual files separately for detailed metrics
3. Smaller dots in visualizations for better clarity
4. Complete TP/FP/TN/FN analysis per file

Key improvements:
- Single training phase using all available normal data
- Individual file testing with preserved trained models
- Enhanced visualization with smaller markers
- Comprehensive metrics per file and mean across all files
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


class OptimizedFlowCytometryPipeline:
    """
    Optimized pipeline with single training phase and individual file testing.
    """
    
    def __init__(self):
        # Data storage
        self.normal_files = []
        self.noise_files = []
        self.all_normal_data = None
        self.trained_models = {}
        self.ensemble_methods = {}
        
        # Scalers and preprocessing
        self.scalers = {'standard': StandardScaler(), 'robust': RobustScaler(), 'minmax': MinMaxScaler()}
        self.best_scaler = None
        self.fitted_scaler = None
        
        # Results storage
        self.training_times = {}
        self.individual_results = []
        self.mean_performance = {}
        
    def load_data_by_files(self):
        """Load normal and noise files separately."""
        print("============================================================")
        print("OPTIMIZED FLOW CYTOMETRY PIPELINE")
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
        
        print(f"Loaded {len(self.normal_files)} normal files and {len(self.noise_files)} noise files")
        
        if not self.normal_files or not self.noise_files:
            raise ValueError("No data files found. Please ensure FCS files are available.")
    
    def apply_polygonal_filter(self, data):
        """Apply polygonal filtering using FL1-FL2 coordinates."""
        # Polygonal coordinates (log scale): [[4.2, 0], [4.2, 3.2], [6.7, 5.9], [6.7, 0.0]]
        polygon_coords = np.array([
            [10**4.2, 10**0.0],    # [15849, 1]
            [10**4.2, 10**3.2],    # [15849, 1585]
            [10**6.7, 10**5.9],    # [5011872, 794328]
            [10**6.7, 10**0.0]     # [5011872, 1]
        ])
        
        # Create Path object for point-in-polygon test
        polygon = Path(polygon_coords)
        
        # Extract FL1 and FL2 coordinates
        points = np.column_stack([data['FL1'].values, data['FL2'].values])
        
        # Test which points are inside the polygon
        inside_mask = polygon.contains_points(points)
        
        print(f"  Polygonal filter: {inside_mask.sum()}/{len(data)} events retained ({inside_mask.mean()*100:.1f}%)")
        
        return data[inside_mask].copy()
    
    def prepare_training_data(self):
        """Combine all normal data for training and apply filtering."""
        print("\n2. PREPARING COMBINED TRAINING DATA")
        
        # Apply polygonal filtering to all normal files
        filtered_normal_files = []
        for normal_data in self.normal_files:
            filtered_data = self.apply_polygonal_filter(normal_data)
            if len(filtered_data) > 0:
                filtered_normal_files.append(filtered_data)
                print(f"  {normal_data['source_file'].iloc[0]}: {len(filtered_data)} events after filtering")
        
        # Combine all filtered normal data
        if filtered_normal_files:
            self.all_normal_data = pd.concat(filtered_normal_files, ignore_index=True)
            print(f"\nCombined training data: {len(self.all_normal_data)} events from {len(filtered_normal_files)} files")
        else:
            raise ValueError("No data available after polygonal filtering")
    
    def train_all_algorithms(self):
        """Train all algorithms ONCE on the combined normal data."""
        print("\n3. TRAINING ALL ALGORITHMS (SINGLE TRAINING PHASE)")
        print("=" * 60)
        
        # Prepare features
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        if 'TIME' in self.all_normal_data.columns:
            feature_cols.append('TIME')
        
        X_train = self.all_normal_data[feature_cols].fillna(0).values
        
        # Select best scaler
        print("Selecting optimal scaler...")
        best_score = 0
        for scaler_name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X_train)
                # Quick evaluation with Isolation Forest
                iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
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
        self.fitted_scaler = self.scalers[self.best_scaler]
        X_scaled = self.fitted_scaler.fit_transform(X_train)
        
        # Estimate contamination rate (conservative)
        total_noise_events = sum(len(self.apply_polygonal_filter(noise_data)) for noise_data in self.noise_files)
        contamination_rate = min(0.2, total_noise_events / (len(self.all_normal_data) + total_noise_events))
        print(f"Estimated contamination rate: {contamination_rate:.3f}")
        
        # Algorithm configurations with optimizations
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
        
        # Train DBSCAN with hyperparameter tuning
        print("Training DBSCAN with hyperparameter optimization...")
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
                    score = n_clusters / (1 + n_noise/len(X_scaled))
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
            except:
                continue
        
        self.trained_models['dbscan'] = DBSCAN(**best_dbscan_params)
        self.training_times['dbscan'] = time.time() - start_time
        print(f"  ✓ DBSCAN trained with params {best_dbscan_params} in {self.training_times['dbscan']:.2f}s")
        
        # Train other algorithms
        for alg_name, config in algorithms_config.items():
            print(f"Training {alg_name}...")
            start_time = time.time()
            try:
                model = config['model'](**config['params'])
                model.fit(X_scaled)
                self.trained_models[alg_name] = model
                self.training_times[alg_name] = time.time() - start_time
                print(f"  ✓ {alg_name} trained in {self.training_times[alg_name]:.2f}s")
            except Exception as e:
                print(f"  ✗ {alg_name} training failed: {e}")
                self.training_times[alg_name] = 0
        
        # Train multiple Bayesian methods
        print("Training Advanced Bayesian Methods...")
        bayesian_methods = [
            'temporal_cooccurrence',
            'bayesian_mixture', 
            'bayesian_ridge',
            'dirichlet_process',
            'change_point_detection',
            'ensemble_bayesian',
            'bayesian_network'
        ]
        
        for method in bayesian_methods:
            start_time = time.time()
            try:
                feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                if X_train.shape[1] >= len(feature_names):
                    X_df = pd.DataFrame(X_train, columns=feature_names)
                else:
                    X_df = pd.DataFrame(X_train)
                    X_df['TIME'] = np.arange(len(X_train))
                    X_df.columns = feature_names[:X_train.shape[1]] + ['TIME']
                
                bayesian_denoiser = BayesianTemporalDenoiser(
                    method=method,
                    time_window=1000,
                    n_components_max=min(15, len(X_train) // 50)  # Adaptive components
                )
                bayesian_denoiser.fit(X_df)
                
                self.trained_models[f'bayesian_{method}'] = bayesian_denoiser
                self.training_times[f'bayesian_{method}'] = time.time() - start_time
                print(f"  ✓ Bayesian {method} trained in {self.training_times[f'bayesian_{method}']:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Bayesian {method} training failed: {e}")
                self.training_times[f'bayesian_{method}'] = 0
        
        # Create ensemble methods
        self._create_ensemble_methods()
        
        total_training_time = sum(self.training_times.values())
        print(f"\nTotal training time: {total_training_time:.2f}s")
        print(f"Successfully trained {len([t for t in self.training_times.values() if t > 0])} algorithms")
    
    def _create_ensemble_methods(self):
        """Create ensemble methods."""
        self.ensemble_methods = {
            'majority_voting': self._majority_voting,
            'weighted_voting': self._weighted_voting,
            'conservative_ensemble': self._conservative_ensemble
        }
    
    def test_individual_files(self):
        """Test each file combination separately using pre-trained models."""
        print("\n4. TESTING INDIVIDUAL FILE COMBINATIONS")
        print("=" * 60)
        
        self.individual_results = []
        
        # Test each combination of normal and noise files
        for normal_data in self.normal_files:
            for noise_data in self.noise_files:
                normal_file = normal_data['source_file'].iloc[0]
                noise_file = noise_data['source_file'].iloc[0]
                print(f"\nTesting: {normal_file} vs {noise_file}")
                
                # Apply polygonal filtering
                normal_filtered = self.apply_polygonal_filter(normal_data)
                noise_filtered = self.apply_polygonal_filter(noise_data)
                
                print(f"  Normal events after filtering: {len(normal_filtered)}")
                print(f"  Noise events after filtering: {len(noise_filtered)}")
                
                if len(normal_filtered) == 0 or len(noise_filtered) == 0:
                    print("  ⚠ Skipping - insufficient data after filtering")
                    continue
                
                # Prepare test data
                feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
                if 'TIME' in normal_filtered.columns and 'TIME' in noise_filtered.columns:
                    feature_cols.append('TIME')
                
                # Combine test data
                combined_data = pd.concat([normal_filtered, noise_filtered], ignore_index=True)
                combined_data['true_label'] = [0] * len(normal_filtered) + [1] * len(noise_filtered)
                
                X_test = combined_data[feature_cols].fillna(0).values
                y_true = combined_data['true_label'].values
                
                # Scale test data using fitted scaler
                X_test_scaled = self.fitted_scaler.transform(X_test)
                
                # Test each algorithm using pre-trained models
                file_results = {
                    'normal_file': normal_file,
                    'noise_file': noise_file,
                    'normal_events': len(normal_filtered),
                    'noise_events': len(noise_filtered),
                    'algorithms': {}
                }
                
                # Test individual algorithms
                for alg_name, model in self.trained_models.items():
                    if self.training_times.get(alg_name, 0) > 0:  # Only test successfully trained models
                        try:
                            results = self._evaluate_algorithm(model, X_test, X_test_scaled, y_true, alg_name)
                            file_results['algorithms'][alg_name] = results
                            print(f"    {alg_name:20}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, "
                                  f"TP={results['tp']:2d}, FP={results['fp']:2d}, TN={results['tn']:2d}, FN={results['fn']:2d}")
                        except Exception as e:
                            print(f"    {alg_name:20}: Failed - {e}")
                            file_results['algorithms'][alg_name] = self._get_default_metrics()
                
                # Compute all predictions once for ensemble methods (optimization)
                predictions_cache = self._compute_all_predictions(X_test, X_test_scaled)
                
                # Test ensemble methods using cached predictions
                for ensemble_name, ensemble_func in self.ensemble_methods.items():
                    try:
                        y_pred = ensemble_func(X_test, X_test_scaled, predictions_cache)
                        results = self._calculate_metrics(y_true, y_pred)
                        file_results['algorithms'][ensemble_name] = results
                        print(f"    {ensemble_name:20}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, "
                              f"TP={results['tp']:2d}, FP={results['fp']:2d}, TN={results['tn']:2d}, FN={results['fn']:2d}")
                    except Exception as e:
                        print(f"    {ensemble_name:20}: Failed - {e}")
                        file_results['algorithms'][ensemble_name] = self._get_default_metrics()
                
                self.individual_results.append(file_results)
        
        # Calculate mean performance across all combinations
        self._calculate_mean_performance()
    
    def _evaluate_algorithm(self, model, X_test, X_test_scaled, y_true, alg_name):
        """Evaluate a single algorithm using pre-trained model."""
        if alg_name == 'dbscan':
            # DBSCAN needs to be fit on test data
            labels = model.fit_predict(X_test_scaled)
            y_pred = (labels == -1).astype(int)
        elif alg_name == 'gaussian_mixture':
            # Gaussian Mixture needs to be fit on test data
            labels = model.fit_predict(X_test_scaled)
            unique_labels, counts = np.unique(labels, return_counts=True)
            noise_cluster = unique_labels[np.argmin(counts)]
            y_pred = (labels == noise_cluster).astype(int)
        elif alg_name.startswith('bayesian_'):
            # Handle all Bayesian methods
            feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
            X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
            if 'TIME' not in X_df.columns:
                X_df['TIME'] = np.arange(len(X_test))
            y_pred = model.predict(X_df)
        else:
            # Outlier detection algorithms (trained models)
            y_pred = model.predict(X_test_scaled)
            y_pred = (y_pred == -1).astype(int)
        
        return self._calculate_metrics(y_true, y_pred)
    
    def _compute_all_predictions(self, X_test, X_test_scaled):
        """Compute predictions for all algorithms once and cache them."""
        predictions_cache = {}
        
        for alg_name, model in self.trained_models.items():
            if self.training_times.get(alg_name, 0) == 0:  # Skip failed models
                continue
                
            try:
                if alg_name == 'dbscan':
                    labels = model.fit_predict(X_test_scaled)
                    predictions_cache[alg_name] = (labels == -1).astype(int)
                elif alg_name == 'gaussian_mixture':
                    labels = model.fit_predict(X_test_scaled)
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    noise_cluster = unique_labels[np.argmin(counts)]
                    predictions_cache[alg_name] = (labels == noise_cluster).astype(int)
                elif alg_name == 'bayesian_temporal':
                    feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                    X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                    if 'TIME' not in X_df.columns:
                        X_df['TIME'] = np.arange(len(X_test))
                    predictions_cache[alg_name] = model.predict(X_df)
                elif alg_name.startswith('bayesian_'):
                    # Handle all other Bayesian methods
                    feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
                    X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
                    if 'TIME' not in X_df.columns:
                        X_df['TIME'] = np.arange(len(X_test))
                    predictions_cache[alg_name] = model.predict(X_df)
                else:
                    pred = model.predict(X_test_scaled)
                    predictions_cache[alg_name] = (pred == -1).astype(int)
            except Exception as e:
                print(f"Warning: Failed to compute predictions for {alg_name}: {e}")
                continue
        
        return predictions_cache
    
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
    
    def _majority_voting(self, X_test, X_test_scaled, predictions_cache=None):
        """Majority voting ensemble using pre-trained models."""
        if predictions_cache is None:
            predictions_cache = self._compute_all_predictions(X_test, X_test_scaled)
        
        predictions = []
        for alg_name in predictions_cache:
            predictions.append(predictions_cache[alg_name])
        
        if len(predictions) > 0:
            return (np.mean(predictions, axis=0) > 0.5).astype(int)
        else:
            return np.zeros(len(X_test), dtype=int)
    
    def _weighted_voting(self, X_test, X_test_scaled, predictions_cache=None):
        """Weighted voting ensemble based on training performance."""
        if predictions_cache is None:
            predictions_cache = self._compute_all_predictions(X_test, X_test_scaled)
        
        predictions = []
        weights = []
        
        # Enhanced weights based on algorithm type and expected performance
        algorithm_weights = {
            'isolation_forest': 1.0,
            'lof': 1.0,
            'one_class_svm': 0.8,
            'elliptic_envelope': 0.9,
            'gaussian_mixture': 1.1,
            'dbscan': 1.2,
            'bayesian_temporal_cooccurrence': 1.3,
            'bayesian_bayesian_mixture': 1.4,
            'bayesian_bayesian_ridge': 1.2,
            'bayesian_dirichlet_process': 1.5,
            'bayesian_change_point_detection': 1.1,
            'bayesian_ensemble_bayesian': 1.6,  # Highest weight for ensemble
            'bayesian_bayesian_network': 1.4
        }
        
        for alg_name in predictions_cache:
            predictions.append(predictions_cache[alg_name])
            weights.append(algorithm_weights.get(alg_name, 1.0))
        
        if len(predictions) > 0:
            weighted_predictions = np.average(predictions, axis=0, weights=weights)
            return (weighted_predictions > 0.5).astype(int)
        else:
            return np.zeros(len(X_test), dtype=int)
    
    def _conservative_ensemble(self, X_test, X_test_scaled, predictions_cache=None):
        """Conservative ensemble requiring multiple algorithms to agree."""
        if predictions_cache is None:
            predictions_cache = self._compute_all_predictions(X_test, X_test_scaled)
        
        predictions = []
        for alg_name in predictions_cache:
            predictions.append(predictions_cache[alg_name])
        
        if len(predictions) > 0:
            # Require at least 2/3 of algorithms to agree on noise classification
            vote_sum = np.sum(predictions, axis=0)
            threshold = max(2, len(predictions) * 0.67)
            return (vote_sum >= threshold).astype(int)
        else:
            return np.zeros(len(X_test), dtype=int)
    
    def _calculate_mean_performance(self):
        """Calculate mean performance across all file combinations."""
        if not self.individual_results:
            return
        
        print("\n5. MEAN PERFORMANCE ACROSS ALL FILE COMBINATIONS")
        print("=" * 60)
        
        # Collect all algorithm names
        all_algorithms = set()
        for result in self.individual_results:
            all_algorithms.update(result['algorithms'].keys())
        
        # Calculate means and standard deviations
        self.mean_performance = {}
        
        for alg_name in all_algorithms:
            metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 
                      'tp': [], 'fp': [], 'tn': [], 'fn': []}
            
            for result in self.individual_results:
                if alg_name in result['algorithms']:
                    alg_results = result['algorithms'][alg_name]
                    for metric in metrics:
                        metrics[metric].append(alg_results[metric])
            
            # Calculate means and stds
            self.mean_performance[alg_name] = {}
            for metric, values in metrics.items():
                if values:
                    self.mean_performance[alg_name][f'{metric}_mean'] = np.mean(values)
                    self.mean_performance[alg_name][f'{metric}_std'] = np.std(values)
                else:
                    self.mean_performance[alg_name][f'{metric}_mean'] = 0.0
                    self.mean_performance[alg_name][f'{metric}_std'] = 0.0
        
        # Display results
        print(f"{'Algorithm':<20} {'Accuracy':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
        print("-" * 80)
        
        # Sort by F1-score
        sorted_algorithms = sorted(all_algorithms, 
                                 key=lambda x: self.mean_performance[x]['f1_score_mean'], 
                                 reverse=True)
        
        for alg_name in sorted_algorithms:
            perf = self.mean_performance[alg_name]
            print(f"{alg_name:<20} "
                  f"{perf['accuracy_mean']:.3f}±{perf['accuracy_std']:.3f} "
                  f"{perf['f1_score_mean']:.3f}±{perf['f1_score_std']:.3f} "
                  f"{perf['tp_mean']:.1f}±{perf['tp_std']:.1f} "
                  f"{perf['fp_mean']:.1f}±{perf['fp_std']:.1f} "
                  f"{perf['tn_mean']:.1f}±{perf['tn_std']:.1f} "
                  f"{perf['fn_mean']:.1f}±{perf['fn_std']:.1f}")
    
    def create_enhanced_visualizations(self):
        """Create comprehensive visualizations with smaller dots."""
        if not self.individual_results:
            print("No results to visualize")
            return
        
        print("\n6. CREATING ENHANCED VISUALIZATIONS")
        
        # Use the last test result for visualization
        last_result = self.individual_results[-1]
        
        # Prepare combined data for visualization
        normal_file_name = last_result['normal_file']
        noise_file_name = last_result['noise_file']
        
        # Find the corresponding data
        normal_data = None
        noise_data = None
        
        for data in self.normal_files:
            if data['source_file'].iloc[0] == normal_file_name:
                normal_data = self.apply_polygonal_filter(data)
                break
        
        for data in self.noise_files:
            if data['source_file'].iloc[0] == noise_file_name:
                noise_data = self.apply_polygonal_filter(data)
                break
        
        if normal_data is None or noise_data is None:
            print("Could not find data for visualization")
            return
        
        # Combine data
        combined_data = pd.concat([normal_data, noise_data], ignore_index=True)
        combined_data['true_label'] = [0] * len(normal_data) + [1] * len(noise_data)
        
        # Prepare data for all algorithms
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        if 'TIME' in combined_data.columns:
            feature_cols.append('TIME')
        
        X_test = combined_data[feature_cols].fillna(0).values
        y_true = combined_data['true_label'].values
        X_test_scaled = self.fitted_scaler.transform(X_test)
        
        # Compute all predictions once for visualization (optimization)
        predictions_cache = self._compute_all_predictions(X_test, X_test_scaled)
        
        # Create comprehensive visualization
        n_algorithms = len(last_result['algorithms'])
        n_cols = min(4, n_algorithms)
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for idx, (alg_name, alg_results) in enumerate(last_result['algorithms'].items()):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            try:
                # Get predictions for this algorithm from cache
                if alg_name in self.trained_models and self.training_times.get(alg_name, 0) > 0:
                    y_pred = predictions_cache.get(alg_name)
                    if y_pred is None:
                        ax.text(0.5, 0.5, f'Error: No predictions for {alg_name}', transform=ax.transAxes, ha='center')
                        ax.set_title(f'{alg_name} - Error')
                        continue
                elif alg_name in self.ensemble_methods:
                    y_pred = self.ensemble_methods[alg_name](X_test, X_test_scaled, predictions_cache)
                else:
                    ax.text(0.5, 0.5, f'Error: Unknown algorithm {alg_name}', transform=ax.transAxes, ha='center')
                    ax.set_title(f'{alg_name} - Error')
                    continue
                
                # Create classification for visualization
                classification = np.zeros(len(y_true))
                classification[(y_true == 1) & (y_pred == 1)] = 1  # TP
                classification[(y_true == 0) & (y_pred == 0)] = 2  # TN
                classification[(y_true == 0) & (y_pred == 1)] = 3  # FP
                classification[(y_true == 1) & (y_pred == 0)] = 4  # FN
                
                # Plot with smaller dots
                colors = ['gray', 'green', 'blue', 'red', 'orange']
                labels = ['Unknown', 'TP', 'TN', 'FP', 'FN']
                
                for class_idx in range(1, 5):
                    mask = classification == class_idx
                    if mask.any():
                        ax.scatter(combined_data.loc[mask, 'FL1'], combined_data.loc[mask, 'FL2'],
                                 c=colors[class_idx], label=labels[class_idx], alpha=0.7, s=8)  # Smaller dots (s=8)
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('FL1')
                ax.set_ylabel('FL2')
                ax.set_title(f'{alg_name}\nAcc: {alg_results["accuracy"]:.3f}, F1: {alg_results["f1_score"]:.3f}')
                ax.legend(markerscale=1.5, fontsize=8)  # Larger legend markers
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, ha='center')
                ax.set_title(f'{alg_name} - Error')
        
        plt.tight_layout()
        plt.savefig('optimized_algorithm_visualizations.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to 'optimized_algorithm_visualizations.png'")
        plt.close()  # Close the figure to free memory
        
        # Create performance comparison
        self._create_performance_comparison()
    
    def _get_predictions_for_visualization(self, model, X_test, X_test_scaled, alg_name):
        """Get predictions for visualization."""
        if alg_name == 'dbscan':
            labels = model.fit_predict(X_test_scaled)
            return (labels == -1).astype(int)
        elif alg_name == 'gaussian_mixture':
            labels = model.fit_predict(X_test_scaled)
            unique_labels, counts = np.unique(labels, return_counts=True)
            noise_cluster = unique_labels[np.argmin(counts)]
            return (labels == noise_cluster).astype(int)
        elif alg_name.startswith('bayesian_'):
            feature_names = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
            X_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
            if 'TIME' not in X_df.columns:
                X_df['TIME'] = np.arange(len(X_test))
            return model.predict(X_df)
        else:
            y_pred = model.predict(X_test_scaled)
            return (y_pred == -1).astype(int)
    
    def _create_performance_comparison(self):
        """Create performance comparison chart."""
        if not self.mean_performance:
            return
        
        # Extract data for plotting
        algorithms = list(self.mean_performance.keys())
        accuracies = [self.mean_performance[alg]['accuracy_mean'] for alg in algorithms]
        f1_scores = [self.mean_performance[alg]['f1_score_mean'] for alg in algorithms]
        
        # Sort by F1-score
        sorted_indices = np.argsort(f1_scores)[::-1]
        algorithms = [algorithms[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
        
        plt.xlabel('Algorithms')
        plt.ylabel('Performance Score')
        plt.title('Mean Performance Comparison Across All File Combinations')
        plt.xticks(x, algorithms, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('optimized_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Performance comparison saved to 'optimized_performance_comparison.png'")
        plt.close()  # Close the figure to free memory
    
    def save_models(self, directory='trained_models'):
        """Save trained models."""
        os.makedirs(directory, exist_ok=True)
        
        # Save scaler
        with open(f'{directory}/scaler.pkl', 'wb') as f:
            pickle.dump(self.fitted_scaler, f)
        
        # Save models
        for alg_name, model in self.trained_models.items():
            if self.training_times.get(alg_name, 0) > 0:
                with open(f'{directory}/{alg_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        print(f"Models saved to {directory}/")
    
    def run_complete_pipeline(self):
        """Run the complete optimized pipeline."""
        try:
            # Load data
            self.load_data_by_files()
            
            # Prepare combined training data
            self.prepare_training_data()
            
            # Train all algorithms ONCE
            self.train_all_algorithms()
            
            # Test on individual files
            self.test_individual_files()
            
            # Create visualizations
            self.create_enhanced_visualizations()
            
            # Save models
            self.save_models()
            
            print("\n" + "="*60)
            print("OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Key Improvements:")
            print("✓ Single training phase on all combined normal data")
            print("✓ Individual file testing with detailed TP/FP/TN/FN metrics")
            print("✓ Enhanced visualizations with smaller, clearer dots")
            print("✓ Mean performance calculation across all file combinations")
            print("✓ Optimized training times (especially SVM)")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    pipeline = OptimizedFlowCytometryPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()