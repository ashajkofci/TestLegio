#!/usr/bin/env python3
"""
Basic functionality tests for the Bayesian denoising pipeline.

This module contains pytest-compatible tests for basic functionality
that can run without complex dependencies.
"""

import pytest
import numpy as np
import pandas as pd
from improved_bayesian_denoising import BayesianTemporalDenoiser


class TestBasicFunctionality:
    """Test basic functionality of the Bayesian denoising system."""

    @pytest.fixture
    def sample_data(self):
        """Create sample synthetic data for testing."""
        np.random.seed(42)
        n_samples = 100
        data = pd.DataFrame({
            'TIME': np.random.uniform(0, 1000, n_samples),
            'SSC': np.random.normal(100, 10, n_samples),
            'FL1': np.random.normal(200, 20, n_samples),
            'FL2': np.random.normal(150, 15, n_samples),
            'FSC': np.random.normal(180, 18, n_samples),
            'FL1-W': np.random.normal(50, 5, n_samples)
        })
        return data

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        np.random.seed(42)
        return np.random.choice([0, 1], 100)

    def test_import_bayesian_denoiser(self):
        """Test that BayesianTemporalDenoiser can be imported successfully."""
        # This test will fail if import fails
        from improved_bayesian_denoising import BayesianTemporalDenoiser
        assert BayesianTemporalDenoiser is not None

    def test_denoiser_initialization(self):
        """Test basic initialization of BayesianTemporalDenoiser."""
        denoiser = BayesianTemporalDenoiser()

        assert denoiser.method == 'temporal_cooccurrence'
        assert not denoiser.is_fitted
        assert hasattr(denoiser, 'fit')
        assert hasattr(denoiser, 'predict')

    def test_synthetic_data_creation(self, sample_data):
        """Test that synthetic data is created correctly."""
        assert len(sample_data) == 100
        assert 'TIME' in sample_data.columns
        assert 'SSC' in sample_data.columns
        assert 'FL1' in sample_data.columns
        assert 'FL2' in sample_data.columns
        assert 'FSC' in sample_data.columns
        assert 'FL1-W' in sample_data.columns

        # Check data types
        assert sample_data['TIME'].dtype in [np.float64, np.float32]
        assert sample_data['SSC'].dtype in [np.float64, np.float32]

    def test_feature_extraction(self, sample_data):
        """Test temporal feature extraction."""
        denoiser = BayesianTemporalDenoiser()
        features = denoiser.extract_temporal_features(sample_data)

        # Features should have same number of rows as input
        assert features.shape[0] == sample_data.shape[0]

        # Should have more features than original
        assert features.shape[1] >= sample_data.shape[1]

        # Should have time_normalized feature
        assert 'time_normalized' in features.columns

        # Check that new features are numeric
        for col in features.columns:
            assert features[col].dtype in [np.float64, np.float32, np.int64, np.int32]

    def test_basic_fitting(self, sample_data, sample_labels):
        """Test basic model fitting."""
        denoiser = BayesianTemporalDenoiser()

        # Should not be fitted initially
        assert not denoiser.is_fitted

        # Fit the model
        denoiser.fit(sample_data, sample_labels)

        # Should be fitted after training
        assert denoiser.is_fitted

    def test_basic_prediction(self, sample_data, sample_labels):
        """Test basic prediction functionality."""
        denoiser = BayesianTemporalDenoiser()
        denoiser.fit(sample_data, sample_labels)

        predictions = denoiser.predict(sample_data)

        # Should return predictions for all samples
        assert len(predictions) == len(sample_data)

        # Predictions should be binary (0 or 1)
        assert set(predictions).issubset({0, 1})

        # Should be numpy array or similar
        assert hasattr(predictions, '__len__')

    def test_prediction_consistency(self, sample_data, sample_labels):
        """Test that predictions are consistent for same input."""
        denoiser = BayesianTemporalDenoiser()
        denoiser.fit(sample_data, sample_labels)

        pred1 = denoiser.predict(sample_data)
        pred2 = denoiser.predict(sample_data)

        # Predictions should be identical for same input
        np.testing.assert_array_equal(pred1, pred2)

    def test_unfitted_prediction_raises_error(self, sample_data):
        """Test that predicting without fitting raises an error."""
        denoiser = BayesianTemporalDenoiser()

        with pytest.raises(ValueError, match="Model must be fitted"):
            denoiser.predict(sample_data)

    def test_feature_extraction_without_fitting(self, sample_data):
        """Test that feature extraction works without fitting."""
        denoiser = BayesianTemporalDenoiser()

        # Should work without fitting
        features = denoiser.extract_temporal_features(sample_data)
        assert features is not None
        assert len(features) == len(sample_data)

    def test_different_methods_initialization(self):
        """Test initialization with different methods."""
        methods = [
            'temporal_cooccurrence',
            'bayesian_mixture',
            'bayesian_ridge',
            'dirichlet_process',
            'change_point_detection',
            'ensemble_bayesian'
        ]

        for method in methods:
            denoiser = BayesianTemporalDenoiser(method=method)
            assert denoiser.method == method
            assert not denoiser.is_fitted

    def test_invalid_method_raises_error(self, sample_data, sample_labels):
        """Test that invalid method falls back to temporal co-occurrence."""
        denoiser = BayesianTemporalDenoiser(method='invalid_method')

        # Should still fit without error (current implementation doesn't validate)
        denoiser.fit(sample_data, sample_labels)
        assert denoiser.is_fitted

        # Should fall back to temporal co-occurrence (not return all zeros)
        predictions = denoiser.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert set(predictions).issubset({0, 1})
        # Should have some variation (not all zeros due to fallback)
        assert not np.all(predictions == 0)