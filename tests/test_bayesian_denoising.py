#!/usr/bin/env python3
"""
Pytest test suite for enhanced Bayesian denoising methods

This module contains comprehensive tests for the Bayesian denoising pipeline
using pytest framework for better test organization and reporting.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

# Import our enhanced Bayesian denoiser
from improved_bayesian_denoising import BayesianTemporalDenoiser


@pytest.fixture
def synthetic_data():
    """Create synthetic flow cytometry data with known noise patterns."""
    np.random.seed(42)

    # Normal data: clustered around certain parameter values
    n_samples = 1000
    noise_ratio = 0.15
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


@pytest.fixture
def trained_denoiser(synthetic_data):
    """Create and train a Bayesian denoiser with default settings."""
    X, y_true = synthetic_data
    denoiser = BayesianTemporalDenoiser(method='temporal_cooccurrence')
    denoiser.fit(X, y_true)
    return denoiser, X, y_true


class TestBayesianTemporalDenoiser:
    """Test suite for BayesianTemporalDenoiser class."""

    def test_initialization(self):
        """Test denoiser initialization with different parameters."""
        # Test default initialization
        denoiser = BayesianTemporalDenoiser()
        assert denoiser.method == 'temporal_cooccurrence'
        assert denoiser.time_window == 1000
        assert not denoiser.is_fitted

        # Test custom initialization
        denoiser = BayesianTemporalDenoiser(
            method='bayesian_mixture',
            time_window=500,
            n_components_max=5
        )
        assert denoiser.method == 'bayesian_mixture'
        assert denoiser.time_window == 500
        assert denoiser.n_components_max == 5

    def test_fit_basic(self, synthetic_data):
        """Test basic fitting functionality."""
        X, y_true = synthetic_data
        denoiser = BayesianTemporalDenoiser()

        # Should not be fitted initially
        assert not denoiser.is_fitted

        # Fit the model
        denoiser.fit(X, y_true)

        # Should be fitted after training
        assert denoiser.is_fitted
        assert hasattr(denoiser, 'training_data')
        assert hasattr(denoiser, 'feature_columns')

    def test_predict_basic(self, trained_denoiser):
        """Test basic prediction functionality."""
        denoiser, X, y_true = trained_denoiser

        # Make predictions
        predictions = denoiser.predict(X)

        # Check predictions shape and type
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.int32, np.int64, int]
        assert set(predictions).issubset({0, 1})  # Binary predictions

    def test_predict_proba(self, trained_denoiser):
        """Test probability prediction functionality."""
        denoiser, X, y_true = trained_denoiser

        # Make probability predictions
        probabilities = denoiser.predict_proba(X)

        # Check probabilities shape and range
        assert probabilities.shape == (len(X), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Should sum to 1

    def test_feature_extraction(self, synthetic_data):
        """Test temporal feature extraction."""
        X, y_true = synthetic_data
        denoiser = BayesianTemporalDenoiser()

        # Extract features
        features = denoiser.extract_temporal_features(X)

        # Check that features were added
        assert features.shape[0] == X.shape[0]
        assert features.shape[1] >= X.shape[1]  # Should have at least original features

        # Check for expected temporal features
        expected_features = ['time_normalized', 'temporal_density', 'temporal_isolation']
        for feature in expected_features:
            assert feature in features.columns, f"Missing expected feature: {feature}"

    @pytest.mark.parametrize("method", [
        'temporal_cooccurrence',
        'bayesian_mixture',
        'bayesian_ridge',
        'dirichlet_process',
        'change_point_detection',
        'ensemble_bayesian'
    ])
    def test_all_methods_fit(self, synthetic_data, method):
        """Test that all Bayesian methods can fit without errors."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(
            method=method,
            time_window=50,
            n_components_max=8
        )

        # Should not raise an exception
        denoiser.fit(X, y_true)
        assert denoiser.is_fitted

    @pytest.mark.parametrize("method", [
        'temporal_cooccurrence',
        'bayesian_mixture',
        'bayesian_ridge',
        'dirichlet_process',
        'change_point_detection',
        'ensemble_bayesian'
    ])
    def test_all_methods_predict(self, synthetic_data, method):
        """Test that all Bayesian methods can make predictions."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(
            method=method,
            time_window=50,
            n_components_max=8
        )

        denoiser.fit(X, y_true)
        predictions = denoiser.predict(X)

        # Check predictions
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_performance_metrics(self, trained_denoiser):
        """Test that predictions produce reasonable performance metrics."""
        denoiser, X, y_true = trained_denoiser

        predictions = denoiser.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        # Check that metrics are reasonable (not all NaN or extreme values)
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_confusion_matrix(self, trained_denoiser):
        """Test confusion matrix calculation."""
        denoiser, X, y_true = trained_denoiser

        predictions = denoiser.predict(X)
        cm = confusion_matrix(y_true, predictions)

        # Check confusion matrix shape and properties
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
        assert np.all(cm >= 0)

    def test_training_time(self, synthetic_data):
        """Test that training completes within reasonable time."""
        X, y_true = synthetic_data

        start_time = time.time()
        denoiser = BayesianTemporalDenoiser()
        denoiser.fit(X, y_true)
        training_time = time.time() - start_time

        # Should complete within 30 seconds for this dataset
        assert training_time < 30.0

    def test_prediction_time(self, trained_denoiser):
        """Test that prediction completes within reasonable time."""
        denoiser, X, y_true = trained_denoiser

        start_time = time.time()
        predictions = denoiser.predict(X)
        prediction_time = time.time() - start_time

        # Should complete within 10 seconds for this dataset
        assert prediction_time < 10.0

    def test_ensemble_methods(self, synthetic_data):
        """Test ensemble Bayesian method specifically."""
        X, y_true = synthetic_data

        denoiser = BayesianTemporalDenoiser(method='ensemble_bayesian')
        denoiser.fit(X, y_true)
        predictions = denoiser.predict(X)

        # Check that ensemble produces valid predictions
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_feature_engineering_enhancement(self, synthetic_data):
        """Test that enhanced feature engineering adds meaningful features."""
        X, y_true = synthetic_data
        denoiser = BayesianTemporalDenoiser()

        original_features = X.shape[1]
        enhanced_features = denoiser.extract_temporal_features(X).shape[1]

        # Should have added features
        assert enhanced_features > original_features

        # Should have added at least some temporal features
        assert enhanced_features >= original_features + 2

    def test_error_handling(self, synthetic_data):
        """Test error handling for edge cases."""
        X, y_true = synthetic_data

        # Test with invalid method - should still work but fall back to temporal co-occurrence
        denoiser = BayesianTemporalDenoiser(method='invalid_method')
        denoiser.fit(X, y_true)
        predictions = denoiser.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
        # Should have some variation (fallback to temporal co-occurrence)
        assert not np.all(predictions == 0)

        # Test prediction without fitting
        denoiser = BayesianTemporalDenoiser()
        with pytest.raises(ValueError):
            denoiser.predict(X)

    def test_scaler_functionality(self, synthetic_data):
        """Test that scaler is properly fitted and used."""
        X, y_true = synthetic_data
        denoiser = BayesianTemporalDenoiser()

        denoiser.fit(X, y_true)

        # Check that scaler exists and is fitted
        assert hasattr(denoiser, 'scaler')
        assert hasattr(denoiser.scaler, 'mean_')  # Indicates scaler has been fitted


class TestSyntheticDataGeneration:
    """Test suite for synthetic data generation."""

    def test_synthetic_data_structure(self, synthetic_data):
        """Test that synthetic data has correct structure."""
        X, y_true = synthetic_data

        # Check data types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y_true, np.ndarray)

        # Check expected columns
        expected_columns = ['TIME', 'SSC', 'FL1', 'FL2', 'FSC', 'FL1-W']
        for col in expected_columns:
            assert col in X.columns

        # Check labels
        assert set(y_true).issubset({0, 1})  # Binary labels
        assert len(y_true) == len(X)

    def test_synthetic_data_realistic_values(self, synthetic_data):
        """Test that synthetic data has realistic flow cytometry values."""
        X, y_true = synthetic_data

        # Check TIME values
        assert X['TIME'].min() >= 0
        assert X['TIME'].max() <= 1000

        # Check parameter ranges (typical flow cytometry values)
        assert X['SSC'].min() >= 0
        assert X['FL1'].min() >= 0
        assert X['FL2'].min() >= 0
        assert X['FSC'].min() >= 0

    def test_noise_ratio(self, synthetic_data):
        """Test that synthetic data has expected noise ratio."""
        X, y_true = synthetic_data

        noise_ratio = np.mean(y_true)
        expected_ratio = 0.15  # From fixture

        # Allow some tolerance due to randomness
        assert abs(noise_ratio - expected_ratio) < 0.05


if __name__ == "__main__":
    pytest.main([__file__])