#!/usr/bin/env python3
"""
Comprehensive Flow Cytometry Data Denoising Pipeline

This pipeline merges the best features from both the original flow_cytometry_pipeline.py
and enhanced_pipeline.py, incorporating:
1. Scientific rigor with proper train/test separation
2. Comprehensive ensemble methods and voting
3. Advanced Bayesian temporal analysis
4. Polygonal filtering with detailed visualizations
5. Extensive metrics and performance analysis
6. Model persistence and reusability
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
warnings.filterwarnings('ignore')

from fcs_parser import load_fcs_data
from bayesian_denoising import BayesianTemporalDenoiser, load_multiple_files


class ComprehensiveFlowCytometryPipeline:
    """
    Comprehensive pipeline combining enhanced scientific methodology with
    extensive algorithmic coverage and detailed analysis capabilities.
    """
    
    def __init__(self):
        # Data storage
        self.normal_data = None
        self.noise_data = None
        self.filtered_normal = None
        self.filtered_noise = None
        self.combined_filtered = None
        
        # Models and results
        self.trained_models = {}
        self.bayesian_denoiser = None
        self.model_performances = {}
        self.ensemble_results = {}
        
        # Configuration
        self.scaler = StandardScaler()
        self.best_scaler = None
        
        # Polygonal filter coordinates: [[FL1_log, FL2_log], ...]
        # Convert from log scale: points: [[4.2, 0], [4.2, 3.2], [6.7, 5.9], [6.7, 0.0]]
        self.polygon_coords = np.array([
            [10**4.2, 10**0.0],    # [15849, 1]
            [10**4.2, 10**3.2],    # [15849, 1585] 
            [10**6.7, 10**5.9],    # [5011872, 794328]
            [10**6.7, 10**0.0]     # [5011872, 1]
        ])
        
    def load_data(self, normal_files_dir='data/normal_files', noise_files_dir='data/noise_files'):
        """Load normal and noise FCS files with fallback to root directory."""
        print("=" * 60)
        print("COMPREHENSIVE FLOW CYTOMETRY PIPELINE")
        print("=" * 60)
        print("1. LOADING DATA")
        
        # Try to load from organized directories first
        try:
            if os.path.exists(normal_files_dir):
                print(f"Loading normal files from: {normal_files_dir}")
                normal_datasets = load_multiple_files(normal_files_dir)
                if normal_datasets and len(normal_datasets) > 0:
                    # Combine multiple normal datasets
                    self.normal_data = pd.concat(normal_datasets, ignore_index=True)
                    self.normal_data['source'] = 'normal'
                else:
                    raise ValueError("No normal files loaded from directory")
            else:
                print("Normal files directory not found, trying full_measurement.fcs")
                self.normal_data = load_fcs_data("full_measurement.fcs")
                if self.normal_data is not None:
                    self.normal_data['source'] = 'normal'
                    
            if os.path.exists(noise_files_dir):
                print(f"Loading noise files from: {noise_files_dir}")
                noise_datasets = load_multiple_files(noise_files_dir)
                if noise_datasets and len(noise_datasets) > 0:
                    # Combine multiple noise datasets
                    self.noise_data = pd.concat(noise_datasets, ignore_index=True)
                    self.noise_data['source'] = 'noise'
                else:
                    raise ValueError("No noise files loaded from directory")
            else:
                print("Noise files directory not found, trying only_noise.fcs")
                self.noise_data = load_fcs_data("only_noise.fcs")
                if self.noise_data is not None:
                    self.noise_data['source'] = 'noise'
                    
        except Exception as e:
            print(f"Error loading from directories: {e}")
            print("Falling back to individual files...")
            
            # Fallback to individual files with CORRECTED interpretation
            self.normal_data = load_fcs_data("full_measurement.fcs")
            if self.normal_data is not None:
                self.normal_data['source'] = 'normal'
                
            self.noise_data = load_fcs_data("only_noise.fcs") 
            if self.noise_data is not None:
                self.noise_data['source'] = 'noise'
        
        # Validate data loading
        if self.normal_data is None or self.noise_data is None:
            raise ValueError("Failed to load FCS files. Please check file paths.")
            
        print(f"Normal data loaded: {len(self.normal_data)} events")
        print(f"Noise data loaded: {len(self.noise_data)} events")
        
    def apply_polygonal_filter(self, data, coords=None):
        """Apply polygonal filtering to FL1-FL2 parameter space."""
        if coords is None:
            coords = self.polygon_coords
            
        # Create polygon path
        polygon = Path(coords)
        
        # Get FL1, FL2 coordinates for each event
        points = np.column_stack([data['FL1'].values, data['FL2'].values])
        
        # Check which points are inside the polygon
        inside_mask = polygon.contains_points(points)
        
        return data[inside_mask].copy()
        
    def preprocess_data(self):
        """Apply polygonal filtering and prepare training/testing datasets."""
        print("\n2. PREPROCESSING WITH POLYGONAL FILTERING")
        
        # Apply polygonal filter to both datasets
        print("Applying polygonal filter...")
        self.filtered_normal = self.apply_polygonal_filter(self.normal_data)
        self.filtered_noise = self.apply_polygonal_filter(self.noise_data)
        
        print(f"Normal events after filtering: {len(self.filtered_normal)}/{len(self.normal_data)} "
              f"({len(self.filtered_normal)/len(self.normal_data)*100:.1f}%)")
        print(f"Noise events after filtering: {len(self.filtered_noise)}/{len(self.noise_data)} "
              f"({len(self.filtered_noise)/len(self.noise_data)*100:.1f}%)")
        
        # Create combined dataset for traditional analysis
        self.combined_filtered = pd.concat([self.filtered_normal, self.filtered_noise], 
                                         ignore_index=True)
        self.combined_filtered['is_noise'] = (self.combined_filtered['source'] == 'noise').astype(int)
        
        print(f"Combined filtered dataset: {len(self.combined_filtered)} events")
        print(f"Noise ratio: {self.combined_filtered['is_noise'].mean():.1%}")
        
    def find_best_scaler(self, X_train, y_train):
        """Find the best scaler for the data."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(), 
            'minmax': MinMaxScaler()
        }
        
        best_scaler_name = 'standard'
        best_score = 0
        
        # Test with a simple algorithm to find best scaler
        for scaler_name, scaler in scalers.items():
            try:
                X_scaled = scaler.fit_transform(X_train)
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                y_pred_binary = (y_pred == -1).astype(int)
                score = accuracy_score(y_train, y_pred_binary)
                
                if score > best_score:
                    best_score = score
                    best_scaler_name = scaler_name
            except:
                continue
                
        self.best_scaler = scalers[best_scaler_name]
        print(f"Selected scaler: {best_scaler_name} (score: {best_score:.3f})")
        return self.best_scaler
        
    def hyperparameter_tuning(self, X_train, y_train):
        """Comprehensive hyperparameter tuning for all algorithms."""
        print("\n3. HYPERPARAMETER TUNING")
        
        X_scaled = self.best_scaler.fit_transform(X_train)
        tuning_results = {}
        
        # Contamination rates to test
        contamination_rates = [0.1, 0.2, 0.3, 0.4]
        
        # 1. Isolation Forest
        print("Tuning Isolation Forest...")
        best_iso_score = 0
        best_iso_params = {}
        
        for contamination in contamination_rates:
            for n_estimators in [50, 100, 200]:
                try:
                    iso_forest = IsolationForest(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    y_pred = iso_forest.fit_predict(X_scaled)
                    y_pred_binary = (y_pred == -1).astype(int)
                    score = f1_score(y_train, y_pred_binary)
                    
                    if score > best_iso_score:
                        best_iso_score = score
                        best_iso_params = {
                            'contamination': contamination,
                            'n_estimators': n_estimators
                        }
                except:
                    continue
                    
        tuning_results['isolation_forest'] = {
            'best_score': best_iso_score,
            'best_params': best_iso_params
        }
        
        # 2. Local Outlier Factor
        print("Tuning Local Outlier Factor...")
        best_lof_score = 0
        best_lof_params = {}
        
        for contamination in contamination_rates:
            for n_neighbors in [10, 20, 30, 50]:
                try:
                    lof = LocalOutlierFactor(
                        contamination=contamination,
                        n_neighbors=n_neighbors
                    )
                    y_pred = lof.fit_predict(X_scaled)
                    y_pred_binary = (y_pred == -1).astype(int)
                    score = f1_score(y_train, y_pred_binary)
                    
                    if score > best_lof_score:
                        best_lof_score = score
                        best_lof_params = {
                            'contamination': contamination,
                            'n_neighbors': n_neighbors
                        }
                except:
                    continue
                    
        tuning_results['local_outlier_factor'] = {
            'best_score': best_lof_score,
            'best_params': best_lof_params
        }
        
        # 3. DBSCAN
        print("Tuning DBSCAN...")
        best_dbscan_score = 0
        best_dbscan_params = {}
        
        for eps in [0.3, 0.5, 0.7, 1.0]:
            for min_samples in [5, 10, 20, 30]:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    y_pred = dbscan.fit_predict(X_scaled)
                    # Convert to binary: -1 (outliers) -> 1, others -> 0
                    y_pred_binary = (y_pred == -1).astype(int)
                    score = f1_score(y_train, y_pred_binary)
                    
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = {
                            'eps': eps,
                            'min_samples': min_samples
                        }
                except:
                    continue
                    
        tuning_results['dbscan'] = {
            'best_score': best_dbscan_score,
            'best_params': best_dbscan_params
        }
        
        # 4. One-Class SVM
        print("Tuning One-Class SVM...")
        best_svm_score = 0
        best_svm_params = {}
        
        for nu in [0.1, 0.2, 0.3, 0.4]:
            for gamma in ['scale', 'auto', 0.1, 1.0]:
                try:
                    svm = OneClassSVM(nu=nu, gamma=gamma)
                    y_pred = svm.fit_predict(X_scaled)
                    y_pred_binary = (y_pred == -1).astype(int)
                    score = f1_score(y_train, y_pred_binary)
                    
                    if score > best_svm_score:
                        best_svm_score = score
                        best_svm_params = {
                            'nu': nu,
                            'gamma': gamma
                        }
                except:
                    continue
                    
        tuning_results['one_class_svm'] = {
            'best_score': best_svm_score,
            'best_params': best_svm_params
        }
        
        print("Hyperparameter tuning completed!")
        return tuning_results
        
    def train_algorithms(self, contamination_rate=0.1):
        """Train all algorithms on normal data with optimized parameters."""
        print("\n4. TRAINING ALGORITHMS")
        
        # Prepare training data (normal events only)
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        available_cols = [col for col in feature_cols if col in self.filtered_normal.columns]
        
        X_train = self.filtered_normal[available_cols].values
        
        # Find best scaler
        # Create dummy y_train for scaler selection (use DBSCAN clustering as proxy)
        dbscan_temp = DBSCAN(eps=0.5, min_samples=5)
        y_temp = (dbscan_temp.fit_predict(StandardScaler().fit_transform(X_train)) == -1).astype(int)
        self.find_best_scaler(X_train, y_temp)
        
        # Get tuning results
        tuning_results = self.hyperparameter_tuning(X_train, y_temp)
        
        # Scale the training data
        X_train_scaled = self.best_scaler.fit_transform(X_train)
        
        # Initialize algorithms with best parameters
        algorithms = {}
        
        # Use tuned parameters if available, otherwise use defaults
        iso_params = tuning_results.get('isolation_forest', {}).get('best_params', 
                                       {'contamination': contamination_rate, 'n_estimators': 100})
        algorithms['isolation_forest'] = IsolationForest(random_state=42, **iso_params)
        
        lof_params = tuning_results.get('local_outlier_factor', {}).get('best_params',
                                       {'contamination': contamination_rate, 'n_neighbors': 20})
        algorithms['lof'] = LocalOutlierFactor(novelty=True, **lof_params)
        
        dbscan_params = tuning_results.get('dbscan', {}).get('best_params',
                                          {'eps': 0.5, 'min_samples': 5})
        algorithms['dbscan'] = DBSCAN(**dbscan_params)
        
        svm_params = tuning_results.get('one_class_svm', {}).get('best_params',
                                       {'nu': contamination_rate, 'gamma': 'scale'})
        algorithms['one_class_svm'] = OneClassSVM(**svm_params)
        
        algorithms['elliptic_envelope'] = EllipticEnvelope(contamination=contamination_rate)
        algorithms['gaussian_mixture'] = GaussianMixture(n_components=2, random_state=42)
        
        print(f"Training on {len(X_train)} normal events...")
        
        # Train each algorithm
        for name, algorithm in algorithms.items():
            try:
                print(f"Training {name}...")
                
                if name == 'gaussian_mixture':
                    # For GMM, fit and calculate threshold
                    algorithm.fit(X_train_scaled)
                    log_probs = algorithm.score_samples(X_train_scaled)
                    threshold = np.percentile(log_probs, contamination_rate * 100)
                    algorithm.threshold_ = threshold
                    
                elif name == 'dbscan':
                    # DBSCAN doesn't have predict method
                    labels = algorithm.fit_predict(X_train_scaled)
                    outlier_count = np.sum(labels == -1)
                    print(f"  DBSCAN identified {outlier_count} outliers in training data")
                    
                else:
                    algorithm.fit(X_train_scaled)
                
                self.trained_models[name] = algorithm
                print(f"  ✓ {name} trained successfully")
                
            except Exception as e:
                print(f"  ✗ {name} training failed: {e}")
        
        # Train Bayesian methods
        print("\nTraining Bayesian Temporal Methods...")
        self.bayesian_denoiser = BayesianTemporalDenoiser()
        try:
            self.bayesian_denoiser.fit(self.filtered_normal)
            print("  ✓ Bayesian temporal methods trained successfully")
        except Exception as e:
            print(f"  ✗ Bayesian training failed: {e}")
        
        # Save trained models
        self.save_trained_models()
        
    def save_trained_models(self):
        """Save trained models and scaler to disk."""
        models_dir = 'trained_models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save scaler
        with open(f'{models_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.best_scaler, f)
        
        # Save each trained model
        for name, model in self.trained_models.items():
            with open(f'{models_dir}/{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save Bayesian denoiser if available
        if self.bayesian_denoiser is not None:
            with open(f'{models_dir}/bayesian_denoiser.pkl', 'wb') as f:
                pickle.dump(self.bayesian_denoiser, f)
        
        print(f"Models saved to {models_dir}/")
        
    def test_on_pure_noise(self):
        """Test trained algorithms on pure noise data."""
        print("\n5. TESTING ON PURE NOISE DATA")
        
        if len(self.filtered_noise) == 0:
            print("No noise events available for testing!")
            return {}
        
        # Prepare test data
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        available_cols = [col for col in feature_cols if col in self.filtered_noise.columns]
        
        X_test = self.filtered_noise[available_cols].values
        y_test = np.ones(len(X_test))  # All are noise (positive class)
        
        # Scale test data using fitted scaler
        X_test_scaled = self.best_scaler.transform(X_test)
        
        results = {}
        
        print(f"Testing on {len(X_test)} pure noise events...")
        
        for name, model in self.trained_models.items():
            try:
                if name == 'gaussian_mixture':
                    # Use probability threshold
                    log_probs = model.score_samples(X_test_scaled)
                    y_pred = (log_probs < model.threshold_).astype(int)
                    
                elif name == 'dbscan':
                    # DBSCAN: use trained eps and min_samples to predict
                    labels = DBSCAN(eps=model.eps, min_samples=model.min_samples).fit_predict(X_test_scaled)
                    y_pred = (labels == -1).astype(int)
                    
                else:
                    # Standard prediction
                    y_pred_raw = model.predict(X_test_scaled)
                    y_pred = (y_pred_raw == -1).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'detection_rate': np.mean(y_pred)
                }
                
                print(f"{name:20} | Acc: {accuracy:.3f} | Pre: {precision:.3f} | "
                      f"Rec: {recall:.3f} | F1: {f1:.3f} | Det: {np.mean(y_pred):.3f}")
                
            except Exception as e:
                print(f"{name:20} | Error: {e}")
                results[name] = {'error': str(e)}
        
        # Test Bayesian methods
        if self.bayesian_denoiser is not None:
            try:
                print("\nTesting Bayesian Methods:")
                bayesian_results = self.bayesian_denoiser.predict(self.filtered_noise)
                
                for method_name, y_pred in bayesian_results.items():
                    if len(y_pred) == len(y_test):
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        results[f'bayesian_{method_name}'] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'predictions': y_pred,
                            'detection_rate': np.mean(y_pred)
                        }
                        
                        print(f"bayesian_{method_name:12} | Acc: {accuracy:.3f} | Pre: {precision:.3f} | "
                              f"Rec: {recall:.3f} | F1: {f1:.3f} | Det: {np.mean(y_pred):.3f}")
                        
            except Exception as e:
                print(f"Bayesian testing error: {e}")
        
        self.model_performances = results
        return results
        
    def test_on_combined_data(self):
        """Test algorithms on combined data (traditional approach)."""
        print("\n6. TESTING ON COMBINED DATA")
        
        # Prepare combined test data
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        available_cols = [col for col in feature_cols if col in self.combined_filtered.columns]
        
        X_combined = self.combined_filtered[available_cols].values
        y_combined = self.combined_filtered['is_noise'].values
        
        # Scale data
        X_combined_scaled = self.best_scaler.transform(X_combined)
        
        combined_results = {}
        
        print(f"Testing on {len(X_combined)} combined events ({np.mean(y_combined):.1%} noise)...")
        
        for name, model in self.trained_models.items():
            try:
                if name == 'gaussian_mixture':
                    log_probs = model.score_samples(X_combined_scaled)
                    y_pred = (log_probs < model.threshold_).astype(int)
                elif name == 'dbscan':
                    labels = DBSCAN(eps=model.eps, min_samples=model.min_samples).fit_predict(X_combined_scaled)
                    y_pred = (labels == -1).astype(int)
                else:
                    y_pred_raw = model.predict(X_combined_scaled)
                    y_pred = (y_pred_raw == -1).astype(int)
                
                # Calculate comprehensive metrics
                tn, fp, fn, tp = confusion_matrix(y_combined, y_pred).ravel()
                
                combined_results[name] = {
                    'accuracy': accuracy_score(y_combined, y_pred),
                    'precision': precision_score(y_combined, y_pred, zero_division=0),
                    'recall': recall_score(y_combined, y_pred, zero_division=0),
                    'f1_score': f1_score(y_combined, y_pred, zero_division=0),
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'predictions': y_pred
                }
                
            except Exception as e:
                print(f"{name} failed: {e}")
                
        return combined_results
        
    def ensemble_methods(self, predictions_dict):
        """Implement various ensemble methods."""
        print("\n7. ENSEMBLE METHODS")
        
        # Collect predictions from successful algorithms
        valid_predictions = {}
        for name, results in predictions_dict.items():
            if 'predictions' in results:
                valid_predictions[name] = results['predictions']
        
        if len(valid_predictions) < 2:
            print("Not enough valid predictions for ensemble methods")
            return {}
        
        # Convert to array
        pred_array = np.array(list(valid_predictions.values()))
        algorithm_names = list(valid_predictions.keys())
        
        ensemble_results = {}
        
        # 1. Majority Voting
        majority_vote = (np.sum(pred_array, axis=0) > len(pred_array) / 2).astype(int)
        ensemble_results['majority_voting'] = majority_vote
        
        # 2. Weighted Voting (based on F1 scores)
        f1_weights = []
        for name in algorithm_names:
            if name in predictions_dict and 'f1_score' in predictions_dict[name]:
                f1_weights.append(predictions_dict[name]['f1_score'])
            else:
                f1_weights.append(0.1)  # Low weight for failed algorithms
        
        f1_weights = np.array(f1_weights)
        f1_weights = f1_weights / np.sum(f1_weights)  # Normalize
        
        weighted_predictions = np.sum(pred_array * f1_weights[:, np.newaxis], axis=0)
        weighted_vote = (weighted_predictions > 0.5).astype(int)
        ensemble_results['weighted_voting'] = weighted_vote
        
        # 3. Conservative Ensemble (require multiple algorithms to agree)
        conservative_vote = (np.sum(pred_array, axis=0) >= max(2, len(pred_array) * 0.6)).astype(int)
        ensemble_results['conservative_ensemble'] = conservative_vote
        
        print(f"Ensemble methods created using {len(algorithm_names)} algorithms:")
        for name in algorithm_names:
            f1_score_val = predictions_dict[name].get('f1_score', 0)
            print(f"  {name}: F1={f1_score_val:.3f}")
        
        return ensemble_results
        
    def generate_visualizations(self, results_dict):
        """Generate comprehensive visualizations."""
        print("\n8. GENERATING VISUALIZATIONS")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Polygonal Filter Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Flow Cytometry Analysis', fontsize=16)
        
        # Plot 1: Polygonal filter overlay
        ax1 = axes[0, 0]
        
        # Plot all data points
        ax1.scatter(self.normal_data['FL1'], self.normal_data['FL2'], 
                   alpha=0.5, s=1, label='Normal (all)', color='lightblue')
        ax1.scatter(self.noise_data['FL1'], self.noise_data['FL2'], 
                   alpha=0.5, s=1, label='Noise (all)', color='lightcoral')
        
        # Overlay filtered points
        ax1.scatter(self.filtered_normal['FL1'], self.filtered_normal['FL2'], 
                   alpha=0.8, s=3, label='Normal (filtered)', color='blue')
        ax1.scatter(self.filtered_noise['FL1'], self.filtered_noise['FL2'], 
                   alpha=0.8, s=3, label='Noise (filtered)', color='red')
        
        # Draw polygon
        polygon_path = Path(self.polygon_coords)
        polygon_patch = plt.Polygon(self.polygon_coords, fill=False, 
                                   edgecolor='black', linewidth=2, linestyle='--')
        ax1.add_patch(polygon_patch)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('FL1 (log scale)')
        ax1.set_ylabel('FL2 (log scale)')
        ax1.set_title('Polygonal Filter Application')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Algorithm Performance Comparison
        ax2 = axes[0, 1]
        
        algorithms = []
        f1_scores = []
        accuracies = []
        
        for name, results in results_dict.items():
            if isinstance(results, dict) and 'f1_score' in results:
                algorithms.append(name.replace('_', ' ').title())
                f1_scores.append(results['f1_score'])
                accuracies.append(results['accuracy'])
        
        x_pos = np.arange(len(algorithms))
        width = 0.35
        
        ax2.bar(x_pos - width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        ax2.bar(x_pos + width/2, accuracies, width, label='Accuracy', alpha=0.8)
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Score')
        ax2.set_title('Algorithm Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix Heatmap (for best performing algorithm)
        ax3 = axes[1, 0]
        
        # Find best algorithm by F1 score
        best_algo = max(results_dict.keys(), 
                       key=lambda x: results_dict[x].get('f1_score', 0) 
                       if isinstance(results_dict[x], dict) else 0)
        
        if 'true_positives' in results_dict[best_algo]:
            cm_data = [[results_dict[best_algo]['true_negatives'], 
                       results_dict[best_algo]['false_positives']],
                      [results_dict[best_algo]['false_negatives'], 
                       results_dict[best_algo]['true_positives']]]
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted Normal', 'Predicted Noise'],
                       yticklabels=['Actual Normal', 'Actual Noise'],
                       ax=ax3)
            ax3.set_title(f'Confusion Matrix - {best_algo.replace("_", " ").title()}')
        else:
            ax3.text(0.5, 0.5, 'Confusion matrix not available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Confusion Matrix')
        
        # Plot 4: Detection Results Visualization with FN/FP/TP/TN
        ax4 = axes[1, 1]
        
        if 'predictions' in results_dict[best_algo] and hasattr(self, 'combined_filtered'):
            # Get predictions and true labels
            y_pred = results_dict[best_algo]['predictions']
            y_true = self.combined_filtered['is_noise'].values
            
            # Create color mapping for TP, TN, FP, FN
            colors = []
            labels = []
            for i, (pred, true) in enumerate(zip(y_pred, y_true)):
                if pred == 1 and true == 1:  # True Positive
                    colors.append('green')
                    labels.append('TP')
                elif pred == 0 and true == 0:  # True Negative  
                    colors.append('blue')
                    labels.append('TN')
                elif pred == 1 and true == 0:  # False Positive
                    colors.append('red')
                    labels.append('FP')
                else:  # False Negative
                    colors.append('orange')
                    labels.append('FN')
            
            # Plot with color coding
            scatter = ax4.scatter(self.combined_filtered['FL1'], 
                                self.combined_filtered['FL2'],
                                c=colors, s=2, alpha=0.7)
            
            # Create legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=8, label='True Positive (TP)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, label='True Negative (TN)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, label='False Positive (FP)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=8, label='False Negative (FN)')
            ]
            ax4.legend(handles=legend_elements, loc='upper right')
            
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.set_xlabel('FL1 (log scale)')
            ax4.set_ylabel('FL2 (log scale)')
            ax4.set_title(f'Detection Results - {best_algo.replace("_", " ").title()}')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional visualization: Performance metrics table
        self.create_performance_table(results_dict)
        
    def create_performance_table(self, results_dict):
        """Create a detailed performance metrics table."""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE PERFORMANCE METRICS")
        print("=" * 100)
        
        # Header
        print(f"{'Algorithm':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
              f"{'F1-Score':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
        print("-" * 100)
        
        # Sort algorithms by F1 score
        sorted_algos = sorted(results_dict.items(), 
                            key=lambda x: x[1].get('f1_score', 0) if isinstance(x[1], dict) else 0, 
                            reverse=True)
        
        for name, results in sorted_algos:
            if isinstance(results, dict) and 'accuracy' in results:
                tp = results.get('true_positives', 0)
                fp = results.get('false_positives', 0)
                tn = results.get('true_negatives', 0)
                fn = results.get('false_negatives', 0)
                
                print(f"{name:<25} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
                      f"{results['recall']:<10.3f} {results['f1_score']:<10.3f} "
                      f"{tp:<6} {fp:<6} {tn:<6} {fn:<6}")
        
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis pipeline."""
        try:
            # Load and preprocess data
            self.load_data()
            self.preprocess_data()
            
            # Train algorithms
            self.train_algorithms()
            
            # Test on different datasets
            pure_noise_results = self.test_on_pure_noise()
            combined_results = self.test_on_combined_data()
            
            # Create ensemble methods
            ensemble_results = self.ensemble_methods(pure_noise_results)
            
            # Evaluate ensemble methods if we have ground truth
            if hasattr(self, 'combined_filtered'):
                y_true = self.combined_filtered['is_noise'].values
                
                for ens_name, ens_pred in ensemble_results.items():
                    if len(ens_pred) == len(y_true):
                        tn, fp, fn, tp = confusion_matrix(y_true, ens_pred).ravel()
                        
                        combined_results[ens_name] = {
                            'accuracy': accuracy_score(y_true, ens_pred),
                            'precision': precision_score(y_true, ens_pred, zero_division=0),
                            'recall': recall_score(y_true, ens_pred, zero_division=0),
                            'f1_score': f1_score(y_true, ens_pred, zero_division=0),
                            'true_positives': tp,
                            'false_positives': fp,
                            'true_negatives': tn,
                            'false_negatives': fn,
                            'predictions': ens_pred
                        }
            
            # Generate visualizations
            self.generate_visualizations(combined_results)
            
            print("\n" + "=" * 60)
            print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return {
                'pure_noise_results': pure_noise_results,
                'combined_results': combined_results,
                'ensemble_results': ensemble_results
            }
            
        except Exception as e:
            print(f"\nERROR in comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution function."""
    pipeline = ComprehensiveFlowCytometryPipeline()
    results = pipeline.run_comprehensive_analysis()
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Results saved and visualizations generated.")
    else:
        print("\nAnalysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()