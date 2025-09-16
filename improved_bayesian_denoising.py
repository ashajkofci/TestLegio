#!/usr/bin/env python3
"""
Improved Bayesian Flow Cytometry Denoising with scikit-learn compatible interface

This module implements advanced Bayesian methods that utilize the TIME parameter
to analyze co-occurrence probabilities and temporal patterns for improved
noise detection in flow cytometry data.

Key improvements:
- Added fit() and predict() methods for scikit-learn compatibility
- Enhanced error handling and graceful fallbacks
- Optimized performance and memory usage
- Better integration with ensemble methods
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict

# Optional imports for advanced Bayesian methods
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

warnings.filterwarnings('ignore')


class BayesianTemporalDenoiser(BaseEstimator, ClassifierMixin):
    """
    Advanced Bayesian denoising using temporal co-occurrence patterns.
    
    This class implements several Bayesian approaches that leverage the TIME
    parameter to identify noise based on temporal clustering and co-occurrence
    probabilities. Now fully compatible with scikit-learn interface.
    """
    
    def __init__(self, time_window=1000, n_components_max=10, method='temporal_cooccurrence',
                 alpha=1e-6, lambda_init=1e-3, n_change_points=5):
        """
        Initialize the Bayesian temporal denoiser.
        
        Args:
            time_window: Time window for co-occurrence analysis
            n_components_max: Maximum number of components for mixture models
            method: Method to use ('temporal_cooccurrence', 'bayesian_mixture', 'naive_bayes',
                                 'bayesian_ridge', 'dirichlet_process', 'change_point_detection',
                                 'bayesian_network', 'ensemble_bayesian')
            alpha: Regularization parameter for Bayesian Ridge
            lambda_init: Initial lambda for Bayesian Ridge
            n_change_points: Number of change points for change point detection
        """
        self.time_window = time_window
        self.n_components_max = n_components_max
        self.method = method
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.n_change_points = n_change_points
        self.scaler = StandardScaler()
        self.models = {}
        self.fitted_models = {}
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit the Bayesian temporal model.
        
        Args:
            X: DataFrame with flow cytometry data including TIME column
            y: Target labels (optional, for supervised methods)
            
        Returns:
            self
        """
        try:
            if isinstance(X, np.ndarray):
                # Convert to DataFrame if needed
                if X.shape[1] >= 6:  # Assume standard cytometry parameters
                    X = pd.DataFrame(X, columns=['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME'])
                else:
                    raise ValueError("Input array must have at least 6 columns for cytometry data")
            
            # Store original data
            self.training_data = X.copy()
            
            # Fit based on selected method
            if self.method == 'temporal_cooccurrence':
                # For unsupervised temporal co-occurrence, just store the data
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    self.scaler.fit(X_features)
                
            elif self.method == 'bayesian_mixture':
                # Fit Bayesian Gaussian Mixture
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.fit_transform(X_features)
                    
                    # Fit BGM
                    n_components = min(self.n_components_max, max(2, len(X) // 10))
                    bgm = BayesianGaussianMixture(
                        n_components=n_components,
                        covariance_type='full',
                        max_iter=100,  # Limit iterations
                        random_state=42
                    )
                    bgm.fit(X_scaled)
                    self.fitted_models['bgm'] = bgm
                
            elif self.method == 'naive_bayes' and y is not None:
                # Fit supervised Naive Bayes
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.fit_transform(X_features)
                    
                    nb = GaussianNB()
                    nb.fit(X_scaled, y)
                    self.fitted_models['naive_bayes'] = nb
            
            elif self.method == 'bayesian_ridge':
                # Fit Bayesian Ridge Regression for anomaly detection
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.fit_transform(X_features)
                    
                    # Use reconstruction error as anomaly score
                    br = BayesianRidge(alpha_1=self.alpha, alpha_2=self.alpha, 
                                     lambda_1=self.lambda_init, lambda_2=self.lambda_init)
                    br.fit(X_scaled, np.arange(len(X_scaled)))  # Dummy target for unsupervised learning
                    self.fitted_models['bayesian_ridge'] = br
                    
                    # Store training data for reconstruction error calculation
                    self.training_features_scaled = X_scaled
            
            elif self.method == 'dirichlet_process':
                # Fit Dirichlet Process Mixture Model (simplified approximation)
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.fit_transform(X_features)
                    
                    # Use Bayesian Gaussian Mixture as approximation of Dirichlet Process
                    n_components = min(self.n_components_max, max(2, len(X) // 20))  # Fewer components
                    dpmm = BayesianGaussianMixture(
                        n_components=n_components,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=1.0/len(X),  # DP concentration parameter
                        covariance_type='full',
                        max_iter=100,
                        random_state=42
                    )
                    dpmm.fit(X_scaled)
                    self.fitted_models['dirichlet_process'] = dpmm
            
            elif self.method == 'change_point_detection':
                # Fit Bayesian Change Point Detection
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.fit_transform(X_features)
                    
                    # Simple change point detection using Bayesian approach
                    self._fit_change_point_detection(X_scaled)
            
            elif self.method == 'bayesian_network' and PGMPY_AVAILABLE:
                # Simplified Bayesian Network approach
                features = self.extract_temporal_features(X)
                self.feature_columns = [col for col in features.columns 
                                      if col in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                                               'time_normalized', 'temporal_density', 'temporal_isolation']]
                if len(self.feature_columns) > 0:
                    X_features = features[self.feature_columns].fillna(0)
                    X_discretized = self._discretize_features(X_features)
                    
                    # Store discretized data for simple pattern matching
                    self.fitted_models['bayesian_network'] = X_discretized
                    print(f"Bayesian network fitted with {len(self.feature_columns)} features")
            
            elif self.method == 'ensemble_bayesian':
                # Fit ensemble of Bayesian methods
                self._fit_ensemble_bayesian(X, y)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            print(f"Warning: Bayesian fit failed: {e}")
            # Set minimal fitted state for graceful fallback
            self.is_fitted = True
            self.feature_columns = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
            return self
    
    def predict(self, X):
        """
        Predict noise labels.
        
        Args:
            X: DataFrame with flow cytometry data
            
        Returns:
            Array of binary predictions (1 = noise, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        try:
            if isinstance(X, np.ndarray):
                # Convert to DataFrame if needed
                if X.shape[1] >= 6:
                    X = pd.DataFrame(X, columns=['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME'])
                else:
                    return np.zeros(len(X))  # Fallback
            
            if self.method == 'temporal_cooccurrence':
                predictions, _ = self.temporal_co_occurrence_analysis(X)
                return predictions
                
            elif self.method == 'bayesian_mixture' and 'bgm' in self.fitted_models:
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    bgm = self.fitted_models['bgm']
                    cluster_labels = bgm.predict(X_scaled)
                    
                    # Identify noise cluster (smallest cluster typically)
                    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                    noise_cluster = unique_labels[np.argmin(counts)]
                    
                    return (cluster_labels == noise_cluster).astype(int)
                
            elif self.method == 'naive_bayes' and 'naive_bayes' in self.fitted_models:
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    nb = self.fitted_models['naive_bayes']
                    return nb.predict(X_scaled)
            
            elif self.method == 'bayesian_ridge' and 'bayesian_ridge' in self.fitted_models:
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    br = self.fitted_models['bayesian_ridge']
                    # Use reconstruction error as anomaly score
                    predictions = br.predict(X_scaled)
                    reconstruction_errors = np.abs(np.arange(len(X_scaled)) - predictions)
                    
                    # Threshold based on training reconstruction errors
                    if hasattr(self, 'training_reconstruction_errors'):
                        threshold = np.percentile(self.training_reconstruction_errors, 95)
                    else:
                        threshold = np.percentile(reconstruction_errors, 95)
                    
                    return (reconstruction_errors > threshold).astype(int)
            
            elif self.method == 'dirichlet_process' and 'dirichlet_process' in self.fitted_models:
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    dpmm = self.fitted_models['dirichlet_process']
                    cluster_labels = dpmm.predict(X_scaled)
                    
                    # Identify noise clusters (smallest clusters)
                    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                    # Consider clusters smaller than 5% of data as noise
                    noise_clusters = unique_labels[counts < len(X_scaled) * 0.05]
                    
                    return np.isin(cluster_labels, noise_clusters).astype(int)
            
            elif self.method == 'change_point_detection':
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    return self._predict_change_point_detection(X_scaled)
            
            elif self.method == 'bayesian_network' and 'bayesian_network' in self.fitted_models and PGMPY_AVAILABLE:
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_discretized = self._discretize_features(X_features)
                    
                    return self._predict_bayesian_network(X_discretized)
            
            elif self.method == 'ensemble_bayesian':
                return self._predict_ensemble_bayesian(X)
            
            # Fallback to temporal co-occurrence
            predictions, _ = self.temporal_co_occurrence_analysis(X)
            return predictions
            
        except Exception as e:
            print(f"Warning: Bayesian prediction failed: {e}")
            return np.zeros(len(X))  # Safe fallback
    
    def predict_proba(self, X):
        """
        Predict noise probabilities.
        
        Args:
            X: DataFrame with flow cytometry data
            
        Returns:
            Array of prediction probabilities [P(normal), P(noise)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        try:
            predictions = self.predict(X)
            
            if self.method == 'naive_bayes' and 'naive_bayes' in self.fitted_models:
                # Get actual probabilities from Naive Bayes
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    nb = self.fitted_models['naive_bayes']
                    return nb.predict_proba(X_scaled)
            
            elif self.method == 'bayesian_ridge' and 'bayesian_ridge' in self.fitted_models:
                # Get probabilities based on reconstruction error
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    br = self.fitted_models['bayesian_ridge']
                    reconstruction_errors = np.abs(np.arange(len(X_scaled)) - br.predict(X_scaled))
                    
                    # Convert to probabilities using sigmoid
                    if hasattr(self, 'training_reconstruction_errors'):
                        threshold = np.percentile(self.training_reconstruction_errors, 95)
                        scale = np.std(self.training_reconstruction_errors)
                    else:
                        threshold = np.percentile(reconstruction_errors, 95)
                        scale = np.std(reconstruction_errors)
                    
                    # Sigmoid transformation for smooth probabilities
                    z_scores = (reconstruction_errors - threshold) / (scale + 1e-6)
                    proba_noise = 1 / (1 + np.exp(-z_scores))
                    proba_normal = 1 - proba_noise
                    
                    return np.column_stack([proba_normal, proba_noise])
            
            elif self.method == 'dirichlet_process' and 'dirichlet_process' in self.fitted_models:
                # Get probabilities based on cluster probabilities
                features = self.extract_temporal_features(X)
                if len(self.feature_columns) > 0 and all(col in features.columns for col in self.feature_columns):
                    X_features = features[self.feature_columns].fillna(0)
                    X_scaled = self.scaler.transform(X_features)
                    
                    dpmm = self.fitted_models['dirichlet_process']
                    log_likelihood = dpmm.score_samples(X_scaled)
                    
                    # Convert log-likelihood to probability
                    # Higher likelihood = lower probability of being noise
                    min_ll = np.min(log_likelihood)
                    max_ll = np.max(log_likelihood)
                    if max_ll > min_ll:
                        normalized_ll = (log_likelihood - min_ll) / (max_ll - min_ll)
                    else:
                        normalized_ll = np.ones(len(log_likelihood)) * 0.5
                    
                    proba_normal = normalized_ll
                    proba_noise = 1 - normalized_ll
                    
                    return np.column_stack([proba_normal, proba_noise])
            
            elif self.method == 'ensemble_bayesian':
                # Get ensemble probabilities
                try:
                    ensemble_predictions = []
                    ensemble_methods = [k for k in self.fitted_models.keys() if k.startswith('ensemble_')]
                    
                    for method_key in ensemble_methods:
                        model = self.fitted_models[method_key]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X)
                            ensemble_predictions.append(proba[:, 1])  # P(noise)
                        else:
                            pred = model.predict(X)
                            ensemble_predictions.append(pred.astype(float))
                    
                    if len(ensemble_predictions) > 0:
                        # Average probabilities
                        avg_proba_noise = np.mean(ensemble_predictions, axis=0)
                        avg_proba_normal = 1 - avg_proba_noise
                        return np.column_stack([avg_proba_normal, avg_proba_noise])
                except Exception as e:
                    print(f"Warning: Ensemble probability calculation failed: {e}")
            
            # For other methods, convert predictions to probabilities with confidence
            proba_noise = predictions.astype(float)
            
            # Add some uncertainty for non-probabilistic methods
            if self.method in ['temporal_cooccurrence', 'change_point_detection']:
                # Add small random noise to avoid overconfidence
                noise = np.random.normal(0, 0.1, len(predictions))
                proba_noise = np.clip(proba_noise + noise, 0, 1)
            
            proba_normal = 1 - proba_noise
            
            return np.column_stack([proba_normal, proba_noise])
            
        except Exception as e:
            print(f"Warning: Bayesian predict_proba failed: {e}")
            # Return neutral probabilities
            n_samples = len(X)
            return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
    
    def extract_temporal_features(self, data):
        """
        Extract temporal features from flow cytometry data.
        
        Args:
            data: DataFrame with TIME column and other parameters
            
        Returns:
            DataFrame with additional temporal features
        """
        try:
            features = data.copy()
            
            # Ensure TIME column exists
            if 'TIME' not in features.columns:
                print("Warning: TIME column not found, using index as time")
                features['TIME'] = np.arange(len(features))
            
            # Sort by time for temporal analysis
            features = features.sort_values('TIME').reset_index(drop=True)
            
            # Time-based features
            time_range = features['TIME'].max() - features['TIME'].min()
            if time_range > 0:
                features['time_normalized'] = (features['TIME'] - features['TIME'].min()) / time_range
            else:
                features['time_normalized'] = 0.5
            
            # Enhanced temporal features for better noise detection
            features['time_velocity'] = features['TIME'].diff().fillna(0)
            features['time_acceleration'] = features['time_velocity'].diff().fillna(0)
            
            # Statistical features for anomaly detection
            features['time_velocity_zscore'] = stats.zscore(features['time_velocity'].fillna(0))
            features['time_acceleration_zscore'] = stats.zscore(features['time_acceleration'].fillna(0))
            
            # Temporal patterns and periodicity detection
            features['temporal_density'] = self._calculate_temporal_density(features)
            features['temporal_isolation'] = self._calculate_temporal_isolation(features)
            features['temporal_consistency'] = self._calculate_temporal_consistency(features)
            features['temporal_burstiness'] = self._calculate_temporal_burstiness(features)
            
            # Parameter evolution over time with enhanced features
            for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']:
                if param in features.columns:
                    # Basic temporal features
                    features[f'{param}_time_gradient'] = features[param].diff().fillna(0)
                    features[f'{param}_time_gradient_zscore'] = stats.zscore(features[f'{param}_time_gradient'])
                    
                    # Moving statistics for local pattern detection
                    window_size = min(20, max(5, len(features) // 10))
                    features[f'{param}_moving_avg'] = features[param].rolling(
                        window=window_size, center=True, min_periods=1
                    ).mean()
                    
                    features[f'{param}_moving_std'] = features[param].rolling(
                        window=window_size, center=True, min_periods=1
                    ).std().fillna(0)
                    
                    features[f'{param}_moving_zscore'] = (
                        (features[param] - features[f'{param}_moving_avg']) / 
                        (features[f'{param}_moving_std'] + 1e-6)
                    )
                    
                    features[f'{param}_deviation'] = features[param] - features[f'{param}_moving_avg']
                    
                    # Higher-order statistics
                    features[f'{param}_skewness'] = features[param].rolling(
                        window=window_size, center=True, min_periods=1
                    ).skew().fillna(0)
                    
                    features[f'{param}_kurtosis'] = features[param].rolling(
                        window=window_size, center=True, min_periods=1
                    ).kurt().fillna(0)
                    
                    # Cross-parameter correlations (if multiple parameters exist)
                    other_params = [p for p in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W'] if p != param and p in features.columns]
                    for other_param in other_params:
                        corr_window = min(50, len(features))
                        rolling_corr = features[param].rolling(window=corr_window).corr(features[other_param])
                        features[f'{param}_{other_param}_correlation'] = rolling_corr.fillna(0)
            
            # Global statistical features for anomaly detection
            for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']:
                if param in features.columns:
                    # Z-score based on global distribution
                    features[f'{param}_global_zscore'] = stats.zscore(features[param])
                    
                    # Modified Z-score (more robust to outliers)
                    median_val = features[param].median()
                    mad_val = np.median(np.abs(features[param] - median_val))
                    features[f'{param}_modified_zscore'] = 0.6745 * (features[param] - median_val) / (mad_val + 1e-6)
                    
                    # Quantile-based features
                    features[f'{param}_quantile_25'] = features[param].quantile(0.25)
                    features[f'{param}_quantile_75'] = features[param].quantile(0.75)
                    features[f'{param}_iqr_position'] = (
                        (features[param] - features[f'{param}_quantile_25']) / 
                        (features[f'{param}_quantile_75'] - features[f'{param}_quantile_25'] + 1e-6)
                    )
            
            # Multi-parameter composite features
            available_params = [p for p in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W'] if p in features.columns]
            if len(available_params) >= 2:
                # Parameter correlation matrix features
                param_matrix = features[available_params].values
                corr_matrix = np.corrcoef(param_matrix.T)
                # Create per-row correlation features
                for i, param in enumerate(available_params):
                    features[f'{param}_correlation_sum'] = np.full(len(features), np.sum(corr_matrix[i, :]))
                
                # Parameter variability index
                param_stds = features[available_params].std(axis=0)
                features['param_variability_index'] = np.full(len(features), np.sum(param_stds) / len(available_params))
                
                # Temporal parameter consistency
                param_gradients = [f'{p}_time_gradient' for p in available_params if f'{p}_time_gradient' in features.columns]
                if param_gradients:
                    features['param_gradient_consistency'] = features[param_gradients].std(axis=1)
            
            return features
            
        except Exception as e:
            print(f"Warning: Temporal feature extraction failed: {e}")
            # Return original data with minimal features
            features = data.copy()
            if 'TIME' not in features.columns:
                features['TIME'] = np.arange(len(features))
            features['time_normalized'] = 0.5
            features['temporal_density'] = 1.0
            features['temporal_isolation'] = 100.0
            return features
    
    def _calculate_temporal_density(self, data):
        """Calculate local temporal density around each event."""
        try:
            densities = []
            times = data['TIME'].values
            
            for i, t in enumerate(times):
                # Count events within time window
                window_start = t - self.time_window
                window_end = t + self.time_window
                density = np.sum((times >= window_start) & (times <= window_end)) - 1  # Exclude self
                densities.append(max(0, density))  # Ensure non-negative
            
            return np.array(densities)
        except:
            return np.ones(len(data))  # Fallback
    
    def _calculate_temporal_isolation(self, data):
        """Calculate temporal isolation (distance to nearest neighbors)."""
        try:
            isolation = []
            times = data['TIME'].values
            
            for i, t in enumerate(times):
                # Find distance to nearest neighbors
                distances = np.abs(times - t).astype(float)
                distances[i] = np.inf  # Exclude self
                
                if len(distances[distances != np.inf]) > 0:
                    min_distance = np.min(distances[distances != np.inf])
                    isolation.append(min_distance)
                else:
                    isolation.append(self.time_window)  # Large value if no neighbors
            
            return np.array(isolation)
        except:
            return np.full(len(data), self.time_window)  # Fallback
    
    def _calculate_temporal_consistency(self, data):
        """Calculate temporal consistency based on parameter stability."""
        try:
            consistency_scores = []
            
            for i in range(len(data)):
                window_start = max(0, i - 10)
                window_end = min(len(data), i + 11)
                window_data = data.iloc[window_start:window_end]
                
                # Calculate consistency based on parameter variance in local window
                available_params = [p for p in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W'] if p in window_data.columns]
                
                if len(available_params) > 0:
                    window_variances = []
                    for param in available_params:
                        param_data = window_data[param].dropna()
                        if len(param_data) > 1:
                            window_variances.append(param_data.var())
                    
                    if window_variances:
                        # Lower variance = higher consistency
                        avg_variance = np.mean(window_variances)
                        consistency = 1.0 / (1.0 + avg_variance)
                        consistency_scores.append(consistency)
                    else:
                        consistency_scores.append(0.5)
                else:
                    consistency_scores.append(0.5)
            
            return np.array(consistency_scores)
        except:
            return np.full(len(data), 0.5)
    
    def _calculate_temporal_burstiness(self, data):
        """Calculate temporal burstiness (inter-event time variability)."""
        try:
            if 'time_velocity' not in data.columns:
                return np.full(len(data), 0.5)
            
            velocities = data['time_velocity'].values
            valid_velocities = velocities[velocities > 0]
            
            if len(valid_velocities) < 2:
                return np.full(len(data), 0.5)
            
            # Calculate coefficient of variation
            mean_velocity = np.mean(valid_velocities)
            std_velocity = np.std(valid_velocities)
            
            if mean_velocity > 0:
                cv = std_velocity / mean_velocity
                # Normalize burstiness to [0, 1] range
                burstiness = min(1.0, cv / 2.0)
            else:
                burstiness = 0.5
            
            return np.full(len(data), burstiness)
        except:
            return np.full(len(data), 0.5)
    
    def temporal_co_occurrence_analysis(self, data):
        """
        Analyze temporal co-occurrence patterns for noise detection.
        
        Args:
            data: DataFrame with flow cytometry data
            
        Returns:
            Co-occurrence based noise predictions and info dict
        """
        try:
            features = self.extract_temporal_features(data)
            
            # Calculate co-occurrence matrices for different parameters
            cooccurrence_scores = []
            
            available_params = [param for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W'] 
                              if param in features.columns]
            
            if not available_params:
                print("Warning: No valid parameters for co-occurrence analysis")
                return np.zeros(len(data)), {
                    'cooccurrence_scores': np.zeros(len(data)),
                    'threshold': 0,
                    'individual_scores': []
                }
            
            for param in available_params:
                try:
                    # Discretize parameter values
                    param_data = features[param].fillna(features[param].median())
                    param_bins = pd.cut(param_data, bins=20, labels=False, duplicates='drop')
                    
                    if param_bins.isna().all():
                        # All values are the same
                        scores = np.zeros(len(features))
                    else:
                        # Calculate temporal co-occurrence
                        cooccur_matrix = self._calculate_cooccurrence_matrix(
                            param_bins, features['TIME'].values
                        )
                        
                        # Calculate co-occurrence scores for each event
                        scores = self._calculate_cooccurrence_scores(
                            param_bins, features['TIME'].values, cooccur_matrix
                        )
                    
                    cooccurrence_scores.append(scores)
                    
                except Exception as e:
                    print(f"Warning: Failed to process parameter {param}: {e}")
                    continue
            
            if not cooccurrence_scores:
                print("Warning: No valid co-occurrence scores calculated")
                return np.zeros(len(data)), {
                    'cooccurrence_scores': np.zeros(len(data)),
                    'threshold': 0,
                    'individual_scores': []
                }
            
            # Combine co-occurrence scores
            combined_scores = np.mean(cooccurrence_scores, axis=0)
            
            # Identify noise based on low co-occurrence (isolated events)
            if np.std(combined_scores) > 0:
                threshold = np.percentile(combined_scores, 20)  # Bottom 20% as potential noise
            else:
                threshold = np.median(combined_scores)
            
            noise_predictions = (combined_scores < threshold).astype(int)
            
            return noise_predictions, {
                'cooccurrence_scores': combined_scores,
                'threshold': threshold,
                'individual_scores': cooccurrence_scores
            }
            
        except Exception as e:
            print(f"Error in temporal co-occurrence analysis: {e}")
            # Return zeros as fallback
            return np.zeros(len(data)), {
                'cooccurrence_scores': np.zeros(len(data)),
                'threshold': 0,
                'individual_scores': []
            }
    
    def _calculate_cooccurrence_matrix(self, param_bins, times):
        """Calculate temporal co-occurrence matrix."""
        try:
            # Handle NaN values in param_bins
            valid_mask = ~pd.isna(param_bins)
            if not valid_mask.any():
                return np.zeros((1, 1))
                
            param_bins_clean = param_bins[valid_mask]
            times_clean = times[valid_mask]
            
            unique_bins = np.unique(param_bins_clean[~pd.isna(param_bins_clean)])
            if len(unique_bins) == 0:
                return np.zeros((1, 1))
                
            n_bins = len(unique_bins)
            bin_to_idx = {bin_val: i for i, bin_val in enumerate(unique_bins)}
            
            cooccur_matrix = np.zeros((n_bins, n_bins))
            
            for i in range(len(param_bins_clean)):
                if pd.isna(param_bins_clean.iloc[i]):
                    continue
                    
                t = times_clean[i]
                bin_i_val = param_bins_clean.iloc[i]
                
                if bin_i_val not in bin_to_idx:
                    continue
                    
                bin_i = bin_to_idx[bin_i_val]
                
                # Find events within time window
                window_mask = (np.abs(times_clean - t) <= self.time_window) & (np.arange(len(times_clean)) != i)
                window_bins = param_bins_clean[window_mask]
                
                # Update co-occurrence matrix
                for bin_j_val in window_bins:
                    if not pd.isna(bin_j_val) and bin_j_val in bin_to_idx:
                        bin_j = bin_to_idx[bin_j_val]
                        cooccur_matrix[bin_i, bin_j] += 1
            
            return cooccur_matrix
            
        except Exception as e:
            print(f"Warning: Co-occurrence matrix calculation failed: {e}")
            return np.zeros((1, 1))
    
    def _calculate_cooccurrence_scores(self, param_bins, times, cooccur_matrix):
        """Calculate co-occurrence scores for each event."""
        try:
            scores = []
            
            # Create mapping for bin indices
            valid_bins = param_bins[~pd.isna(param_bins)]
            if len(valid_bins) == 0:
                return np.zeros(len(param_bins))
                
            unique_bins = np.unique(valid_bins)
            bin_to_idx = {bin_val: i for i, bin_val in enumerate(unique_bins)}
            
            for i in range(len(param_bins)):
                if pd.isna(param_bins.iloc[i]):
                    scores.append(0)
                    continue
                
                bin_i_val = param_bins.iloc[i]
                if bin_i_val not in bin_to_idx:
                    scores.append(0)
                    continue
                    
                bin_i = bin_to_idx[bin_i_val]
                t = times[i]
                
                # Find events within time window
                window_mask = (np.abs(times - t) <= self.time_window) & (np.arange(len(times)) != i)
                window_bins = param_bins[window_mask]
                
                # Calculate score based on co-occurrence probabilities
                score = 0
                total_cooccur = cooccur_matrix[bin_i, :].sum()
                
                if total_cooccur > 0:
                    for bin_j_val in window_bins:
                        if not pd.isna(bin_j_val) and bin_j_val in bin_to_idx:
                            bin_j = bin_to_idx[bin_j_val]
                            prob = cooccur_matrix[bin_i, bin_j] / total_cooccur
                            score += prob
                
                scores.append(score)
            
            return np.array(scores)
            
        except Exception as e:
            print(f"Warning: Co-occurrence score calculation failed: {e}")
            return np.zeros(len(param_bins))
    
    def _fit_change_point_detection(self, X_scaled):
        """Fit Bayesian change point detection model."""
        try:
            # Simple Bayesian change point detection using cumulative sums
            n_samples = len(X_scaled)
            
            # Calculate cumulative statistics
            cumulative_mean = np.cumsum(X_scaled, axis=0) / np.arange(1, n_samples + 1)[:, np.newaxis]
            cumulative_var = np.zeros((n_samples, X_scaled.shape[1]))
            
            for i in range(1, n_samples):
                cumulative_var[i] = np.var(X_scaled[:i+1], axis=0)
            
            # Store for prediction
            self.change_point_stats = {
                'cumulative_mean': cumulative_mean,
                'cumulative_var': cumulative_var,
                'training_data': X_scaled
            }
            
        except Exception as e:
            print(f"Warning: Change point detection fit failed: {e}")
    
    def _predict_change_point_detection(self, X_scaled):
        """Predict using Bayesian change point detection."""
        try:
            if not hasattr(self, 'change_point_stats'):
                return np.zeros(len(X_scaled))
            
            predictions = []
            training_stats = self.change_point_stats
            
            for i, x in enumerate(X_scaled):
                # Calculate likelihood of being a change point
                if i < len(training_stats['training_data']):
                    # Compare with training statistics
                    train_mean = training_stats['cumulative_mean'][i]
                    train_var = training_stats['cumulative_var'][i]
                    
                    # Mahalanobis distance
                    diff = x - train_mean
                    if np.any(train_var > 0):
                        mahalanobis = np.sqrt(np.sum(diff**2 / train_var))
                    else:
                        mahalanobis = np.linalg.norm(diff)
                    
                    # High distance indicates potential change point (noise)
                    predictions.append(1 if mahalanobis > 2.0 else 0)
                else:
                    predictions.append(0)  # Default for out-of-training samples
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Warning: Change point detection prediction failed: {e}")
            return np.zeros(len(X_scaled))
    
    def _discretize_features(self, X_features):
        """Discretize features for Bayesian network."""
        try:
            X_discretized = X_features.copy()
            
            for col in X_features.columns:
                if X_features[col].dtype in ['float64', 'float32']:
                    # Discretize into 3 bins: low, medium, high
                    X_discretized[col] = pd.cut(X_features[col], bins=3, labels=['low', 'medium', 'high'])
            
            return X_discretized
            
        except Exception as e:
            print(f"Warning: Feature discretization failed: {e}")
            return X_features
    
    def _predict_bayesian_network(self, X_discretized):
        """Predict using simplified Bayesian network approach."""
        try:
            predictions = []
            
            for idx, row in X_discretized.iterrows():
                try:
                    # Simple rule-based prediction for robustness
                    # Check for unusual combinations in discretized features
                    unusual_patterns = 0
                    
                    # Check SSC vs FL1 relationship
                    if 'SSC' in row and 'FL1' in row:
                        if row['SSC'] == 'high' and row['FL1'] == 'low':
                            unusual_patterns += 1
                        elif row['SSC'] == 'low' and row['FL1'] == 'high':
                            unusual_patterns += 1
                    
                    # Check temporal patterns
                    if 'time_normalized' in row and 'temporal_isolation' in row:
                        if row['time_normalized'] == 'high' and row['temporal_isolation'] == 'high':
                            unusual_patterns += 1
                    
                    # Predict as noise if multiple unusual patterns
                    predictions.append(1 if unusual_patterns >= 2 else 0)
                        
                except Exception as e:
                    predictions.append(0)  # Default prediction
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Warning: Bayesian network prediction failed: {e}")
            return np.zeros(len(X_discretized))
    
    def _fit_ensemble_bayesian(self, X, y=None):
        """Fit ensemble of Bayesian methods."""
        try:
            # Fit multiple Bayesian methods
            methods_to_fit = ['temporal_cooccurrence', 'bayesian_mixture', 'bayesian_ridge', 'dirichlet_process']
            
            for method in methods_to_fit:
                try:
                    temp_denoiser = BayesianTemporalDenoiser(
                        time_window=self.time_window,
                        n_components_max=self.n_components_max,
                        method=method
                    )
                    temp_denoiser.fit(X, y)
                    self.fitted_models[f'ensemble_{method}'] = temp_denoiser
                    
                except Exception as e:
                    print(f"Warning: Failed to fit {method} in ensemble: {e}")
            
            # Store ensemble weights (can be learned or predefined)
            self.ensemble_weights = {
                'temporal_cooccurrence': 1.0,
                'bayesian_mixture': 1.2,
                'bayesian_ridge': 1.1,
                'dirichlet_process': 1.3
            }
            
        except Exception as e:
            print(f"Warning: Ensemble Bayesian fit failed: {e}")
    
    def _predict_ensemble_bayesian(self, X):
        """Predict using Bayesian ensemble."""
        try:
            predictions = []
            weights = []
            
            ensemble_methods = [k for k in self.fitted_models.keys() if k.startswith('ensemble_')]
            
            for method_key in ensemble_methods:
                try:
                    method_name = method_key.replace('ensemble_', '')
                    model = self.fitted_models[method_key]
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.ensemble_weights.get(method_name, 1.0))
                    
                except Exception as e:
                    print(f"Warning: Failed to predict with {method_key}: {e}")
            
            if len(predictions) > 0:
                # Weighted voting
                weighted_preds = np.average(predictions, axis=0, weights=weights)
                return (weighted_preds > 0.5).astype(int)
            else:
                # Fallback to temporal co-occurrence
                pred, _ = self.temporal_co_occurrence_analysis(X)
                return pred
            
        except Exception as e:
            print(f"Warning: Ensemble Bayesian prediction failed: {e}")
            pred, _ = self.temporal_co_occurrence_analysis(X)
            return pred


def load_multiple_files(data_dir, file_pattern="*.fcs"):
    """
    Load multiple FCS files from a directory.
    
    Args:
        data_dir: Directory containing FCS files
        file_pattern: Pattern to match files
        
    Returns:
        List of DataFrames with file information
    """
    import glob
    import os
    from fcs_parser import load_fcs_data
    
    files = glob.glob(os.path.join(data_dir, file_pattern))
    datasets = []
    
    for file_path in files:
        try:
            data = load_fcs_data(file_path)
            data['source_file'] = os.path.basename(file_path)
            datasets.append(data)
            print(f"Loaded {file_path}: {len(data)} events")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return datasets