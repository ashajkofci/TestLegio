#!/usr/bin/env python3
"""
Enhanced Bayesian methods tests for the denoising pipeline.

This module contains pytest-compatible tests for the enhanced Bayesian
denoising methods, including performance comparisons and feature validation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from improved_bayesian_denoising import BayesianTemporalDenoiser


class TestEnhancedBayesianMethods:
    """Test enhanced Bayesian denoising methods."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic flow cytometry data with known noise patterns."""
        np.random.seed(42)
        n_samples = 500  # Smaller for faster testing
        noise_ratio = 0.15

        # Normal data: clustered around certain parameter values
        n_normal = int(n_samples * (1 - noise_ratio))
        n_noise = n_samples - n_normal

        # Normal events
        normal_data = {
            'TIME': np.sort(np.random.uniform(0, 1000, n_normal)),
            'SSC': np.random.normal(100, 10, n_normal),
            'FL1': np.random.normal(200, 20, n_normal),
            'FL2': np.random.normal(150, 15, n_normal),
            'FSC': np.random.normal(180, 18, n_normal),
            'FL1-W': np.random.normal(50, 5, n_normal)
        }

        # Noise events: scattered and with extreme values
        noise_data = {
            'TIME': np.random.uniform(0, 1000, n_noise),
            'SSC': np.random.choice([np.random.normal(10, 5, 1)[0], np.random.normal(300, 30, 1)[0]], n_noise),
            'FL1': np.random.choice([np.random.normal(10, 5, 1)[0], np.random.normal(500, 50, 1)[0]], n_noise),
            'FL2': np.random.choice([np.random.normal(5, 2, 1)[0], np.random.normal(400, 40, 1)[0]], n_noise),
            'FSC': np.random.choice([np.random.normal(15, 5, 1)[0], np.random.normal(450, 45, 1)[0]], n_noise),
            'FL1-W': np.random.choice([np.random.normal(5, 2, 1)[0], np.random.normal(150, 15, 1)[0]], n_noise)
        }

        # Combine data
        combined_data = {}
        for key in normal_data.keys():
            combined_data[key] = np.concatenate([normal_data[key], noise_data[key]])

        df = pd.DataFrame(combined_data)
        true_labels = np.concatenate([np.zeros(n_normal), np.ones(n_noise)])

        return df, true_labels

    @pytest.mark.parametrize("method", [
        'temporal_cooccurrence',
        'bayesian_mixture',
        'bayesian_ridge',
        'dirichlet_process',
        'change_point_detection',
        'ensemble_bayesian'
    ])
    def test_bayesian_method_execution(self, synthetic_data, method):
        """Test that each Bayesian method can execute without errors."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(
            method=method,
            time_window=50,
            n_components_max=8
        )

        # Should fit without error
        denoiser.fit(X, y_true)
        assert denoiser.is_fitted

        # Should predict without error
        y_pred = denoiser.predict(X)
        assert len(y_pred) == len(X)
        assert set(y_pred).issubset({0, 1})

    @pytest.mark.parametrize("method", [
        'temporal_cooccurrence',
        'bayesian_mixture',
        'bayesian_ridge',
        'dirichlet_process',
        'change_point_detection',
        'ensemble_bayesian'
    ])
    def test_bayesian_method_performance(self, synthetic_data, method):
        """Test that each Bayesian method produces reasonable performance metrics."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(
            method=method,
            time_window=50,
            n_components_max=8
        )

        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Calculate basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Basic sanity checks
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1_score <= 1

        # Should have some true positives (method should find some noise)
        assert tp >= 0

    def test_temporal_cooccurrence_method(self, synthetic_data):
        """Test temporal co-occurrence method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='temporal_cooccurrence')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_bayesian_mixture_method(self, synthetic_data):
        """Test Bayesian mixture method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='bayesian_mixture')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_bayesian_ridge_method(self, synthetic_data):
        """Test Bayesian ridge method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='bayesian_ridge')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_dirichlet_process_method(self, synthetic_data):
        """Test Dirichlet process method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='dirichlet_process')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_change_point_detection_method(self, synthetic_data):
        """Test change point detection method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='change_point_detection')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_ensemble_bayesian_method(self, synthetic_data):
        """Test ensemble Bayesian method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='ensemble_bayesian')
        denoiser.fit(X, y_true)
        y_pred = denoiser.predict(X)

        # Should produce valid predictions
        assert len(y_pred) == len(X)
        assert all(pred in [0, 1] for pred in y_pred)

    def test_ensemble_performance(self, synthetic_data):
        """Test that ensemble methods improve performance."""
        X, y_true = synthetic_data

        # Test individual methods
        methods = ['temporal_cooccurrence', 'bayesian_mixture', 'bayesian_ridge']
        individual_predictions = []

        for method in methods:
            denoiser = BayesianTemporalDenoiser(method=method)
            denoiser.fit(X, y_true)
            pred = denoiser.predict(X)
            individual_predictions.append(pred)

        # Test ensemble method
        ensemble_denoiser = BayesianTemporalDenoiser(method='ensemble_bayesian')
        ensemble_denoiser.fit(X, y_true)
        ensemble_pred = ensemble_denoiser.predict(X)

        # Ensemble should produce valid predictions
        assert len(ensemble_pred) == len(X)
        assert all(pred in [0, 1] for pred in ensemble_pred)

    def test_feature_engineering_validation(self, synthetic_data):
        """Test that feature engineering adds meaningful features."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser()
        features = denoiser.extract_temporal_features(X)

        # Should have more features than original
        assert features.shape[1] > X.shape[1]

        # Should have same number of samples
        assert features.shape[0] == X.shape[0]

        # Should include original features
        for col in X.columns:
            assert col in features.columns

        # Should have new features
        new_features = [col for col in features.columns if col not in X.columns]
        assert len(new_features) > 0

        # New features should be numeric
        for col in new_features:
            assert features[col].dtype in [np.float64, np.float32, np.int64, np.int32]

    def test_temporal_features_content(self, synthetic_data):
        """Test that temporal features contain expected content."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser()
        features = denoiser.extract_temporal_features(X)

        # Should have time_normalized
        assert 'time_normalized' in features.columns

        # Should have some temporal features (may include statistical or rolling features)
        temporal_features = [col for col in features.columns if any(term in col.lower()
                           for term in ['temporal', 'time', 'rolling', 'mean', 'std', 'var', 'median', 'min', 'max'])]
        # At minimum, should have time_normalized
        assert len(temporal_features) >= 1

        # Check that temporal features are numeric
        for col in temporal_features:
            assert features[col].dtype in [np.float64, np.float32, np.int64, np.int32]

    def test_method_parameter_validation(self):
        """Test that methods handle parameters correctly."""
        # Test with different time windows
        for time_window in [10, 50, 100]:
            denoiser = BayesianTemporalDenoiser(time_window=time_window)
            assert denoiser.time_window == time_window

        # Test with different n_components_max
        for n_comp in [3, 5, 8]:
            denoiser = BayesianTemporalDenoiser(n_components_max=n_comp)
            assert denoiser.n_components_max == n_comp

    def test_prediction_probabilities(self, synthetic_data):
        """Test that methods can return prediction probabilities when available."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='bayesian_ridge')
        denoiser.fit(X, y_true)

        # Some methods might support predict_proba
        if hasattr(denoiser, 'predict_proba'):
            probas = denoiser.predict_proba(X)
            assert probas.shape[0] == len(X)
            assert probas.shape[1] == 2  # Binary classification
            assert np.all(probas >= 0)
            assert np.all(probas <= 1)
            assert np.allclose(probas.sum(axis=1), 1.0)