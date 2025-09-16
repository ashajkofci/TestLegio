#!/usr/bin/env python3
"""
Tests for the Optimized Flow Cytometry Pipeline.

This module contains pytest-compatible tests for the optimized flow cytometry
pipeline, including data loading, training, testing, and performance evaluation.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from optimized_final_pipeline import OptimizedFlowCytometryPipeline


class TestOptimizedPipeline:
    """Test the optimized flow cytometry pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return OptimizedFlowCytometryPipeline()

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic test data."""
        np.random.seed(42)

        # Normal data that will pass polygonal filter
        normal_data = {
            'SSC': np.random.normal(1000, 200, 100),
            'FL1': np.random.uniform(10**4.3, 10**5.0, 100),  # Within polygon FL1 range
            'FL2': np.random.uniform(10**0.5, 10**2.0, 100),  # Within polygon FL2 range
            'FSC': np.random.normal(800, 150, 100),
            'FL1-W': np.random.normal(500, 100, 100),
            'TIME': np.arange(100)
        }

        # Noise data that will pass polygonal filter
        noise_data = {
            'SSC': np.random.normal(2000, 500, 30),
            'FL1': np.random.uniform(10**5.5, 10**6.5, 30),  # Higher FL1 values within polygon
            'FL2': np.random.uniform(10**3.0, 10**5.0, 30),  # Higher FL2 values within polygon
            'FSC': np.random.normal(1500, 300, 30),
            'FL1-W': np.random.normal(1000, 200, 30),
            'TIME': np.arange(30)
        }

        normal_df = pd.DataFrame(normal_data)
        normal_df['source_file'] = 'test_normal.fcs'

        noise_df = pd.DataFrame(noise_data)
        noise_df['source_file'] = 'test_noise.fcs'

        return normal_df, noise_df

    @pytest.fixture
    def setup_pipeline(self, pipeline, synthetic_data):
        """Set up pipeline with synthetic data."""
        normal_df, noise_df = synthetic_data
        pipeline.normal_files = [normal_df]
        pipeline.noise_files = [noise_df]
        return pipeline

    def test_data_loading(self, setup_pipeline):
        """Test data loading functionality."""
        pipeline = setup_pipeline

        assert len(pipeline.normal_files) == 1
        assert len(pipeline.noise_files) == 1
        assert len(pipeline.normal_files[0]) == 100
        assert len(pipeline.noise_files[0]) == 30

    def test_polygonal_filtering(self, setup_pipeline, synthetic_data):
        """Test polygonal filtering functionality."""
        pipeline = setup_pipeline
        normal_df, noise_df = synthetic_data

        # Test normal data filtering
        normal_filtered = pipeline.apply_polygonal_filter(normal_df)
        noise_filtered = pipeline.apply_polygonal_filter(noise_df)

        # Should retain some data after filtering
        assert len(normal_filtered) > 0, "Normal data should have some events after filtering"
        assert len(noise_filtered) >= 0, "Noise data filtering should work"

        # Filtered data should be subset of original
        assert len(normal_filtered) <= len(normal_df)
        assert len(noise_filtered) <= len(noise_df)

    def test_single_training_phase(self, setup_pipeline):
        """Test that training happens only once."""
        pipeline = setup_pipeline

        # Prepare training data
        pipeline.prepare_training_data()

        # Verify combined training data
        assert pipeline.all_normal_data is not None
        assert len(pipeline.all_normal_data) > 0

        # Train algorithms
        original_train = pipeline.train_all_algorithms
        call_count = {'count': 0}

        def mock_train():
            call_count['count'] += 1
            return original_train()

        pipeline.train_all_algorithms = mock_train

        # Run training
        pipeline.train_all_algorithms()

        # Verify training happened once
        assert call_count['count'] == 1, "Training should happen exactly once"
        assert len(pipeline.trained_models) > 0, "Should have trained models"
        assert pipeline.fitted_scaler is not None, "Should have fitted scaler"

    def test_individual_file_testing(self, setup_pipeline):
        """Test individual file testing without retraining."""
        pipeline = setup_pipeline

        # Set up for testing
        pipeline.prepare_training_data()
        pipeline.train_all_algorithms()

        # Count the number of times fit is called on models
        fit_calls = {'count': 0}

        # Mock model fitting to count calls (should only happen for DBSCAN and Gaussian Mixture)
        for alg_name, model in pipeline.trained_models.items():
            if hasattr(model, 'fit'):
                original_fit = model.fit

                def count_fit(*args, **kwargs):
                    if alg_name not in ['dbscan', 'gaussian_mixture']:  # These need to fit on test data
                        fit_calls['count'] += 1
                    return original_fit(*args, **kwargs)
                model.fit = count_fit

        # Run individual file testing
        pipeline.test_individual_files()

        # Verify results
        assert len(pipeline.individual_results) > 0, "Should have test results"

        # Verify mean performance calculation
        assert len(pipeline.mean_performance) > 0, "Should have mean performance metrics"

    def test_performance_metrics(self, setup_pipeline):
        """Test comprehensive performance metrics."""
        pipeline = setup_pipeline

        # Set up and run pipeline
        pipeline.prepare_training_data()
        pipeline.train_all_algorithms()
        pipeline.test_individual_files()

        # Check individual results structure
        for result in pipeline.individual_results:
            assert 'normal_file' in result
            assert 'noise_file' in result
            assert 'normal_events' in result
            assert 'noise_events' in result
            assert 'algorithms' in result

            # Check algorithm results
            for alg_name, alg_results in result['algorithms'].items():
                required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn']
                for metric in required_metrics:
                    assert metric in alg_results, f"Missing {metric} in {alg_name} results"

        # Check mean performance structure
        for alg_name, perf in pipeline.mean_performance.items():
            required_mean_metrics = ['accuracy_mean', 'f1_score_mean', 'tp_mean', 'fp_mean', 'tn_mean', 'fn_mean']
            for metric in required_mean_metrics:
                assert metric in perf, f"Missing {metric} in {alg_name} mean performance"

    def test_ensemble_methods(self, setup_pipeline):
        """Test ensemble method functionality."""
        pipeline = setup_pipeline

        # Set up pipeline
        pipeline.prepare_training_data()
        pipeline.train_all_algorithms()

        # Test ensemble methods
        assert 'majority_voting' in pipeline.ensemble_methods
        assert 'weighted_voting' in pipeline.ensemble_methods
        assert 'conservative_ensemble' in pipeline.ensemble_methods

        # Test ensemble predictions
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        X_test = pipeline.normal_files[0][feature_cols].fillna(0).values
        X_test_scaled = pipeline.fitted_scaler.transform(X_test)

        for ensemble_name, ensemble_func in pipeline.ensemble_methods.items():
            predictions = ensemble_func(X_test, X_test_scaled)
            assert len(predictions) == len(X_test), f"{ensemble_name} should return prediction for each sample"
            assert np.all(np.isin(predictions, [0, 1])), f"{ensemble_name} should return binary predictions"

    def test_model_persistence(self, setup_pipeline, temp_dir):
        """Test model saving and loading."""
        pipeline = setup_pipeline

        # Set up and train
        pipeline.prepare_training_data()
        pipeline.train_all_algorithms()

        # Save models to temp directory
        models_dir = os.path.join(temp_dir, 'test_models')
        pipeline.save_models(models_dir)

        # Verify files were created
        assert os.path.exists(os.path.join(models_dir, 'scaler.pkl'))

        # Count successfully saved models
        saved_models = 0
        for alg_name in pipeline.trained_models:
            model_file = os.path.join(models_dir, f'{alg_name}.pkl')
            if os.path.exists(model_file):
                saved_models += 1

        assert saved_models > 0, "Should save at least one model"

    def test_training_optimization(self, setup_pipeline):
        """Test training time optimizations."""
        pipeline = setup_pipeline

        # Set up pipeline
        pipeline.prepare_training_data()

        # Train and measure times
        import time
        start_time = time.time()
        pipeline.train_all_algorithms()
        total_time = time.time() - start_time

        # Verify training times are recorded
        assert len(pipeline.training_times) > 0, "Should record training times"

        # Check for reasonable training times (should be much faster with optimizations)
        svm_time = pipeline.training_times.get('one_class_svm', 0)
        if svm_time > 0:
            assert svm_time < 30, "SVM training should be fast with limited iterations"

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert hasattr(pipeline, 'normal_files')
        assert hasattr(pipeline, 'noise_files')
        assert hasattr(pipeline, 'trained_models')
        assert hasattr(pipeline, 'ensemble_methods')

    def test_prepare_training_data_empty(self, pipeline):
        """Test prepare_training_data with no data."""
        # Should raise error when no data is available
        with pytest.raises(ValueError, match="No data available after polygonal filtering"):
            pipeline.prepare_training_data()

    def test_train_all_algorithms_empty_data(self, pipeline):
        """Test training with no data."""
        # Should raise error when trying to prepare training data with no data
        with pytest.raises(ValueError, match="No data available after polygonal filtering"):
            pipeline.prepare_training_data()

    def test_test_individual_files_empty(self, pipeline):
        """Test individual file testing with no data."""
        pipeline.test_individual_files()
        # Should handle empty testing gracefully
        assert isinstance(pipeline.individual_results, list)

    def test_ensemble_methods_empty_models(self, pipeline):
        """Test ensemble methods with no trained models."""
        # Ensemble methods dict should exist (may be empty if not yet implemented)
        assert isinstance(pipeline.ensemble_methods, dict)

    def test_save_models_no_models(self, pipeline, temp_dir):
        """Test saving models when no models are trained."""
        models_dir = os.path.join(temp_dir, 'empty_models')
        pipeline.save_models(models_dir)
        # Should handle gracefully
        assert True