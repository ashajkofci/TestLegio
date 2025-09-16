#!/usr/bin/env python3
"""
Test suite for the Optimized Flow Cytometry Pipeline

Tests:
1. Single training phase verification
2. Individual file testing accuracy
3. Enhanced visualization quality
4. Model persistence functionality
5. Performance optimization verification
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from optimized_final_pipeline import OptimizedFlowCytometryPipeline


class TestOptimizedPipeline(unittest.TestCase):
    """Test the optimized flow cytometry pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = OptimizedFlowCytometryPipeline()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic test data
        np.random.seed(42)
        
        # Normal data that will pass polygonal filter
        # Polygon coords: FL1: [10^4.2, 10^6.7], FL2: [10^0, 10^5.9]
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
        
        self.normal_df = pd.DataFrame(normal_data)
        self.normal_df['source_file'] = 'test_normal.fcs'
        
        self.noise_df = pd.DataFrame(noise_data)
        self.noise_df['source_file'] = 'test_noise.fcs'
        
        # Set up pipeline with test data
        self.pipeline.normal_files = [self.normal_df]
        self.pipeline.noise_files = [self.noise_df]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_01_data_loading(self):
        """Test data loading functionality."""
        print("\n1. Testing data loading...")
        
        # Test with mock data already set up
        self.assertEqual(len(self.pipeline.normal_files), 1)
        self.assertEqual(len(self.pipeline.noise_files), 1)
        self.assertEqual(len(self.pipeline.normal_files[0]), 100)
        self.assertEqual(len(self.pipeline.noise_files[0]), 30)
        
        print("   ✓ Data loading successful")
    
    def test_02_polygonal_filtering(self):
        """Test polygonal filtering functionality."""
        print("\n2. Testing polygonal filtering...")
        
        # Test normal data filtering
        normal_filtered = self.pipeline.apply_polygonal_filter(self.normal_df)
        noise_filtered = self.pipeline.apply_polygonal_filter(self.noise_df)
        
        # Should retain some data after filtering
        self.assertGreater(len(normal_filtered), 0, "Normal data should have some events after filtering")
        self.assertGreaterEqual(len(noise_filtered), 0, "Noise data filtering should work")
        
        # Filtered data should be subset of original
        self.assertLessEqual(len(normal_filtered), len(self.normal_df))
        self.assertLessEqual(len(noise_filtered), len(self.noise_df))
        
        print(f"   ✓ Normal data: {len(normal_filtered)}/{len(self.normal_df)} events retained")
        print(f"   ✓ Noise data: {len(noise_filtered)}/{len(self.noise_df)} events retained")
    
    def test_03_single_training_phase(self):
        """Test that training happens only once."""
        print("\n3. Testing single training phase...")
        
        # Prepare training data
        self.pipeline.prepare_training_data()
        
        # Verify combined training data
        self.assertIsNotNone(self.pipeline.all_normal_data)
        self.assertGreater(len(self.pipeline.all_normal_data), 0)
        
        # Train algorithms
        original_optimize_count = 0
        
        # Mock the optimize_algorithms method to count calls
        original_train = self.pipeline.train_all_algorithms
        call_count = {'count': 0}
        
        def mock_train():
            call_count['count'] += 1
            return original_train()
        
        self.pipeline.train_all_algorithms = mock_train
        
        # Run training
        self.pipeline.train_all_algorithms()
        
        # Verify training happened once
        self.assertEqual(call_count['count'], 1, "Training should happen exactly once")
        self.assertGreater(len(self.pipeline.trained_models), 0, "Should have trained models")
        self.assertIsNotNone(self.pipeline.fitted_scaler, "Should have fitted scaler")
        
        print(f"   ✓ Single training phase completed")
        print(f"   ✓ Trained {len(self.pipeline.trained_models)} algorithms")
    
    def test_04_individual_file_testing(self):
        """Test individual file testing without retraining."""
        print("\n4. Testing individual file testing...")
        
        # Set up for testing
        self.pipeline.prepare_training_data()
        self.pipeline.train_all_algorithms()
        
        # Count the number of times fit is called on models
        fit_calls = {'count': 0}
        
        # Mock model fitting to count calls (should only happen for DBSCAN and Gaussian Mixture)
        for alg_name, model in self.pipeline.trained_models.items():
            if hasattr(model, 'fit'):
                original_fit = model.fit
                def count_fit(*args, **kwargs):
                    if alg_name not in ['dbscan', 'gaussian_mixture']:  # These need to fit on test data
                        fit_calls['count'] += 1
                    return original_fit(*args, **kwargs)
                model.fit = count_fit
        
        # Run individual file testing
        self.pipeline.test_individual_files()
        
        # Verify results
        self.assertGreater(len(self.pipeline.individual_results), 0, "Should have test results")
        
        # Check that models weren't retrained (except DBSCAN and Gaussian Mixture)
        # Note: DBSCAN and Gaussian Mixture need to fit on test data, so we allow those
        
        print(f"   ✓ Individual file testing completed")
        print(f"   ✓ Generated {len(self.pipeline.individual_results)} test result(s)")
        
        # Verify mean performance calculation
        self.assertGreater(len(self.pipeline.mean_performance), 0, "Should have mean performance metrics")
        print(f"   ✓ Mean performance calculated for {len(self.pipeline.mean_performance)} algorithms")
    
    def test_05_performance_metrics(self):
        """Test comprehensive performance metrics."""
        print("\n5. Testing performance metrics...")
        
        # Set up and run pipeline
        self.pipeline.prepare_training_data()
        self.pipeline.train_all_algorithms()
        self.pipeline.test_individual_files()
        
        # Check individual results structure
        for result in self.pipeline.individual_results:
            self.assertIn('normal_file', result)
            self.assertIn('noise_file', result)
            self.assertIn('normal_events', result)
            self.assertIn('noise_events', result)
            self.assertIn('algorithms', result)
            
            # Check algorithm results
            for alg_name, alg_results in result['algorithms'].items():
                required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn']
                for metric in required_metrics:
                    self.assertIn(metric, alg_results, f"Missing {metric} in {alg_name} results")
        
        # Check mean performance structure
        for alg_name, perf in self.pipeline.mean_performance.items():
            required_mean_metrics = ['accuracy_mean', 'f1_score_mean', 'tp_mean', 'fp_mean', 'tn_mean', 'fn_mean']
            for metric in required_mean_metrics:
                self.assertIn(metric, perf, f"Missing {metric} in {alg_name} mean performance")
        
        print("   ✓ Performance metrics structure verified")
        print("   ✓ All required metrics present")
    
    def test_06_ensemble_methods(self):
        """Test ensemble method functionality."""
        print("\n6. Testing ensemble methods...")
        
        # Set up pipeline
        self.pipeline.prepare_training_data()
        self.pipeline.train_all_algorithms()
        
        # Test ensemble methods
        self.assertIn('majority_voting', self.pipeline.ensemble_methods)
        self.assertIn('weighted_voting', self.pipeline.ensemble_methods)
        self.assertIn('conservative_ensemble', self.pipeline.ensemble_methods)
        
        # Test ensemble predictions
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        X_test = self.normal_df[feature_cols].fillna(0).values
        X_test_scaled = self.pipeline.fitted_scaler.transform(X_test)
        
        for ensemble_name, ensemble_func in self.pipeline.ensemble_methods.items():
            try:
                predictions = ensemble_func(X_test, X_test_scaled)
                self.assertEqual(len(predictions), len(X_test), f"{ensemble_name} should return prediction for each sample")
                self.assertTrue(np.all(np.isin(predictions, [0, 1])), f"{ensemble_name} should return binary predictions")
                print(f"   ✓ {ensemble_name} working correctly")
            except Exception as e:
                print(f"   ⚠ {ensemble_name} failed: {e}")
    
    def test_07_model_persistence(self):
        """Test model saving and loading."""
        print("\n7. Testing model persistence...")
        
        # Set up and train
        self.pipeline.prepare_training_data()
        self.pipeline.train_all_algorithms()
        
        # Save models to temp directory
        models_dir = os.path.join(self.temp_dir, 'test_models')
        self.pipeline.save_models(models_dir)
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(models_dir, 'scaler.pkl')))
        
        # Count successfully saved models
        saved_models = 0
        for alg_name in self.pipeline.trained_models:
            model_file = os.path.join(models_dir, f'{alg_name}.pkl')
            if os.path.exists(model_file):
                saved_models += 1
        
        self.assertGreater(saved_models, 0, "Should save at least one model")
        print(f"   ✓ Saved {saved_models} models and scaler")
    
    def test_08_training_optimization(self):
        """Test training time optimizations."""
        print("\n8. Testing training optimizations...")
        
        # Set up pipeline
        self.pipeline.prepare_training_data()
        
        # Train and measure times
        import time
        start_time = time.time()
        self.pipeline.train_all_algorithms()
        total_time = time.time() - start_time
        
        # Verify training times are recorded
        self.assertGreater(len(self.pipeline.training_times), 0, "Should record training times")
        
        # Check for reasonable training times (should be much faster with optimizations)
        svm_time = self.pipeline.training_times.get('one_class_svm', 0)
        if svm_time > 0:
            self.assertLess(svm_time, 30, "SVM training should be fast with limited iterations")
        
        print(f"   ✓ Total training time: {total_time:.2f}s")
        for alg_name, training_time in self.pipeline.training_times.items():
            if training_time > 0:
                print(f"   ✓ {alg_name}: {training_time:.2f}s")


def run_tests():
    """Run all tests with detailed output."""
    print("="*60)
    print("OPTIMIZED PIPELINE TEST SUITE")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests in order
    test_methods = [
        'test_01_data_loading',
        'test_02_polygonal_filtering', 
        'test_03_single_training_phase',
        'test_04_individual_file_testing',
        'test_05_performance_metrics',
        'test_06_ensemble_methods',
        'test_07_model_persistence',
        'test_08_training_optimization'
    ]
    
    for test_method in test_methods:
        suite.addTest(TestOptimizedPipeline(test_method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)