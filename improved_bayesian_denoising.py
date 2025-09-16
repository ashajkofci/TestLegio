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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BayesianTemporalDenoiser(BaseEstimator, ClassifierMixin):
    """
    Advanced Bayesian denoising using temporal co-occurrence patterns.
    
    This class implements several Bayesian approaches that leverage the TIME
    parameter to identify noise based on temporal clustering and co-occurrence
    probabilities. Now fully compatible with scikit-learn interface.
    """
    
    def __init__(self, time_window=1000, n_components_max=10, method='temporal_cooccurrence'):
        """
        Initialize the Bayesian temporal denoiser.
        
        Args:
            time_window: Time window for co-occurrence analysis
            n_components_max: Maximum number of components for mixture models
            method: Method to use ('temporal_cooccurrence', 'bayesian_mixture', 'naive_bayes')
        """
        self.time_window = time_window
        self.n_components_max = n_components_max
        self.method = method
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
            
            # For other methods, convert predictions to probabilities
            proba_noise = predictions.astype(float)
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
            
            features['time_velocity'] = features['TIME'].diff().fillna(0)
            features['time_acceleration'] = features['time_velocity'].diff().fillna(0)
            
            # Temporal co-occurrence features
            features['temporal_density'] = self._calculate_temporal_density(features)
            features['temporal_isolation'] = self._calculate_temporal_isolation(features)
            
            # Parameter evolution over time (only for existing columns)
            for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']:
                if param in features.columns:
                    features[f'{param}_time_gradient'] = features[param].diff().fillna(0)
                    
                    # Moving average with min_periods=1
                    features[f'{param}_moving_avg'] = features[param].rolling(
                        window=min(10, len(features)), center=True, min_periods=1
                    ).mean()
                    
                    features[f'{param}_deviation'] = features[param] - features[f'{param}_moving_avg']
            
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