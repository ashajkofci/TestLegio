#!/usr/bin/env python3
"""
Bayesian Flow Cytometry Denoising with Time-based Co-occurrence Analysis

This module implements advanced Bayesian methods that utilize the TIME parameter
to analyze co-occurrence probabilities and temporal patterns for improved
noise detection in flow cytometry data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class BayesianTemporalDenoiser:
    """
    Advanced Bayesian denoising using temporal co-occurrence patterns.
    
    This class implements several Bayesian approaches that leverage the TIME
    parameter to identify noise based on temporal clustering and co-occurrence
    probabilities.
    """
    
    def __init__(self, time_window=1000, n_components_max=10):
        """
        Initialize the Bayesian temporal denoiser.
        
        Args:
            time_window: Time window for co-occurrence analysis
            n_components_max: Maximum number of components for mixture models
        """
        self.time_window = time_window
        self.n_components_max = n_components_max
        self.scaler = StandardScaler()
        self.models = {}
        
    def extract_temporal_features(self, data):
        """
        Extract temporal features from flow cytometry data.
        
        Args:
            data: DataFrame with TIME column and other parameters
            
        Returns:
            DataFrame with additional temporal features
        """
        features = data.copy()
        
        # Sort by time for temporal analysis
        features = features.sort_values('TIME').reset_index(drop=True)
        
        # Time-based features
        features['time_normalized'] = (features['TIME'] - features['TIME'].min()) / (features['TIME'].max() - features['TIME'].min())
        features['time_velocity'] = features['TIME'].diff().fillna(0)
        features['time_acceleration'] = features['time_velocity'].diff().fillna(0)
        
        # Temporal co-occurrence features
        features['temporal_density'] = self._calculate_temporal_density(features)
        features['temporal_isolation'] = self._calculate_temporal_isolation(features)
        
        # Parameter evolution over time
        for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']:
            features[f'{param}_time_gradient'] = features[param].diff().fillna(0)
            features[f'{param}_moving_avg'] = features[param].rolling(window=10, center=True).mean().fillna(features[param])
            features[f'{param}_deviation'] = features[param] - features[f'{param}_moving_avg']
        
        # Time-binned statistics
        features['time_bin'] = pd.cut(features['TIME'], bins=100, labels=False)
        bin_stats = features.groupby('time_bin')[['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']].agg(['mean', 'std']).fillna(0)
        bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns]
        
        # Map bin statistics back to events
        for col in bin_stats.columns:
            features[f'bin_{col}'] = features['time_bin'].map(bin_stats[col])
        
        return features
    
    def _calculate_temporal_density(self, data):
        """Calculate local temporal density around each event."""
        densities = []
        times = data['TIME'].values
        
        for i, t in enumerate(times):
            # Count events within time window
            window_start = t - self.time_window
            window_end = t + self.time_window
            density = np.sum((times >= window_start) & (times <= window_end)) - 1  # Exclude self
            densities.append(density)
        
        return np.array(densities)
    
    def _calculate_temporal_isolation(self, data):
        """Calculate temporal isolation (distance to nearest neighbors)."""
        isolation = []
        times = data['TIME'].values
        
        for i, t in enumerate(times):
            # Find distance to nearest neighbors
            distances = np.abs(times - t)
            distances[i] = np.inf  # Exclude self
            min_distance = np.min(distances)
            isolation.append(min_distance)
        
        return np.array(isolation)
    
    def bayesian_gaussian_mixture_temporal(self, data, n_components=None):
        """
        Bayesian Gaussian Mixture Model with temporal features.
        
        Args:
            data: DataFrame with flow cytometry data
            n_components: Number of components (auto-selected if None)
            
        Returns:
            Noise predictions and model
        """
        print("Running Bayesian Gaussian Mixture with Temporal Features...")
        
        # Extract temporal features
        features = self.extract_temporal_features(data)
        
        # Select features for modeling
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 
                       'time_normalized', 'temporal_density', 'temporal_isolation']
        
        # Add temporal gradient features
        gradient_cols = [col for col in features.columns if 'gradient' in col]
        feature_cols.extend(gradient_cols[:3])  # Add first 3 gradient features
        
        X = features[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Auto-select number of components if not provided
        if n_components is None:
            n_components = self._select_optimal_components(X_scaled, method='bgm')
        
        # Fit Bayesian Gaussian Mixture
        bgm = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42
        )
        
        cluster_labels = bgm.fit_predict(X_scaled)
        
        # Calculate posterior probabilities
        log_probs = bgm.score_samples(X_scaled)
        posterior_probs = bgm.predict_proba(X_scaled)
        
        # Identify noise cluster based on temporal characteristics
        noise_cluster = self._identify_noise_cluster_temporal(features, cluster_labels, posterior_probs)
        
        # Generate noise predictions
        noise_predictions = (cluster_labels == noise_cluster).astype(int)
        
        # Store model
        self.models['bgm_temporal'] = bgm
        
        return noise_predictions, {
            'cluster_labels': cluster_labels,
            'log_probs': log_probs,
            'posterior_probs': posterior_probs,
            'noise_cluster': noise_cluster,
            'n_components': n_components
        }
    
    def naive_bayes_temporal(self, train_data, test_data=None):
        """
        Naive Bayes classifier with temporal features.
        
        Args:
            train_data: Training data with 'is_noise' labels
            test_data: Test data (if None, uses train_data)
            
        Returns:
            Noise predictions and probabilities
        """
        print("Running Naive Bayes with Temporal Features...")
        
        if test_data is None:
            test_data = train_data
        
        # Extract temporal features
        train_features = self.extract_temporal_features(train_data)
        test_features = self.extract_temporal_features(test_data)
        
        # Select features
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W',
                       'time_normalized', 'temporal_density', 'temporal_isolation']
        
        X_train = train_features[feature_cols].values
        X_test = test_features[feature_cols].values
        y_train = train_data['is_noise'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train_scaled, y_train)
        
        # Predict
        predictions = nb.predict(X_test_scaled)
        probabilities = nb.predict_proba(X_test_scaled)
        
        # Store model
        self.models['naive_bayes_temporal'] = nb
        
        return predictions, {
            'probabilities': probabilities,
            'feature_importance': feature_cols
        }
    
    def temporal_co_occurrence_analysis(self, data):
        """
        Analyze temporal co-occurrence patterns for noise detection.
        
        Args:
            data: DataFrame with flow cytometry data
            
        Returns:
            Co-occurrence based noise predictions
        """
        print("Running Temporal Co-occurrence Analysis...")
        
        features = self.extract_temporal_features(data)
        
        # Calculate co-occurrence matrices for different parameters
        cooccurrence_scores = []
        
        for param in ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']:
            # Discretize parameter values
            param_bins = pd.cut(features[param], bins=20, labels=False)
            
            # Calculate temporal co-occurrence
            cooccur_matrix = self._calculate_cooccurrence_matrix(
                param_bins, features['TIME'].values
            )
            
            # Calculate co-occurrence scores for each event
            scores = self._calculate_cooccurrence_scores(
                param_bins, features['TIME'].values, cooccur_matrix
            )
            cooccurrence_scores.append(scores)
        
        # Combine co-occurrence scores
        combined_scores = np.mean(cooccurrence_scores, axis=0)
        
        # Identify noise based on low co-occurrence (isolated events)
        threshold = np.percentile(combined_scores, 20)  # Bottom 20% as potential noise
        noise_predictions = (combined_scores < threshold).astype(int)
        
        return noise_predictions, {
            'cooccurrence_scores': combined_scores,
            'threshold': threshold,
            'individual_scores': cooccurrence_scores
        }
    
    def _calculate_cooccurrence_matrix(self, param_bins, times):
        """Calculate temporal co-occurrence matrix."""
        n_bins = int(np.nanmax(param_bins)) + 1
        cooccur_matrix = np.zeros((n_bins, n_bins))
        
        for i in range(len(param_bins)):
            if np.isnan(param_bins[i]):
                continue
                
            t = times[i]
            bin_i = int(param_bins[i])
            
            # Find events within time window
            window_mask = (np.abs(times - t) <= self.time_window) & (np.arange(len(times)) != i)
            window_bins = param_bins[window_mask]
            
            # Update co-occurrence matrix
            for bin_j in window_bins:
                if not np.isnan(bin_j):
                    cooccur_matrix[bin_i, int(bin_j)] += 1
        
        return cooccur_matrix
    
    def _calculate_cooccurrence_scores(self, param_bins, times, cooccur_matrix):
        """Calculate co-occurrence scores for each event."""
        scores = []
        
        for i in range(len(param_bins)):
            if np.isnan(param_bins[i]):
                scores.append(0)
                continue
            
            bin_i = int(param_bins[i])
            t = times[i]
            
            # Find events within time window
            window_mask = (np.abs(times - t) <= self.time_window) & (np.arange(len(times)) != i)
            window_bins = param_bins[window_mask]
            
            # Calculate score based on co-occurrence probabilities
            score = 0
            for bin_j in window_bins:
                if not np.isnan(bin_j):
                    if cooccur_matrix[bin_i, :].sum() > 0:
                        prob = cooccur_matrix[bin_i, int(bin_j)] / cooccur_matrix[bin_i, :].sum()
                        score += prob
            
            scores.append(score)
        
        return np.array(scores)
    
    def _identify_noise_cluster_temporal(self, features, cluster_labels, posterior_probs):
        """Identify which cluster represents noise based on temporal characteristics."""
        cluster_stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_data = features[mask]
            
            # Calculate cluster characteristics
            stats = {
                'size': mask.sum(),
                'temporal_density_mean': cluster_data['temporal_density'].mean(),
                'temporal_isolation_mean': cluster_data['temporal_isolation'].mean(),
                'fl1_mean': cluster_data['FL1'].mean(),
                'time_span': cluster_data['TIME'].max() - cluster_data['TIME'].min()
            }
            cluster_stats[cluster_id] = stats
        
        # Identify noise cluster (typically smaller, more isolated, different FL1 characteristics)
        best_noise_cluster = 0
        best_noise_score = 0
        
        for cluster_id, stats in cluster_stats.items():
            # Score based on noise characteristics
            size_score = 1 / (1 + stats['size'] / len(features))  # Smaller clusters
            isolation_score = stats['temporal_isolation_mean'] / (1 + stats['temporal_isolation_mean'])
            density_score = 1 / (1 + stats['temporal_density_mean'])  # Lower density
            
            noise_score = size_score * isolation_score * density_score
            
            if noise_score > best_noise_score:
                best_noise_score = noise_score
                best_noise_cluster = cluster_id
        
        return best_noise_cluster
    
    def _select_optimal_components(self, X, method='bgm', max_components=None):
        """Select optimal number of components using information criteria."""
        if max_components is None:
            max_components = min(self.n_components_max, X.shape[0] // 10)
        
        bic_scores = []
        aic_scores = []
        
        for n in range(1, max_components + 1):
            if method == 'bgm':
                model = BayesianGaussianMixture(n_components=n, random_state=42)
            else:
                from sklearn.mixture import GaussianMixture
                model = GaussianMixture(n_components=n, random_state=42)
            
            try:
                model.fit(X)
                bic_scores.append(model.bic(X))
                aic_scores.append(model.aic(X))
            except:
                bic_scores.append(np.inf)
                aic_scores.append(np.inf)
        
        # Select based on minimum BIC
        optimal_n = np.argmin(bic_scores) + 1
        return optimal_n


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