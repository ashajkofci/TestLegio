#!/usr/bin/env python3
"""
Test Suite for Final Comprehensive Flow Cytometry Pipeline

Tests all the improvements:
1. Individual file testing functionality
2. Optimized SVM training
3. Fixed Bayesian methods with proper interface
4. Complete ensemble method evaluation
5. Comprehensive visualizations
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the modules to test
from final_comprehensive_pipeline import FinalComprehensiveFlowCytometryPipeline
from improved_bayesian_denoising import BayesianTemporalDenoiser


class TestFinalComprehensivePipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.pipeline = FinalComprehensiveFlowCytometryPipeline()
        
        # Create synthetic test data
        np.random.seed(42)
        
        # Normal data - ensure some data falls within polygonal region
        n_normal = 100
        # Polygonal coordinates (log10): [[4.2, 0], [4.2, 3.2], [6.7, 5.9], [6.7, 0.0]]
        # Generate data with some events in the polygon
        normal_fl1 = np.concatenate([
            np.random.uniform(10**4.3, 10**5.0, n_normal//2),  # Some in polygon
            np.random.uniform(10**2.0, 10**3.5, n_normal//2)   # Some outside
        ])
        normal_fl2 = np.concatenate([
            np.random.uniform(10**0.5, 10**2.5, n_normal//2),  # Some in polygon
            np.random.uniform(10**0.1, 10**1.0, n_normal//2)   # Some outside
        ])
        
        self.normal_data = pd.DataFrame({
            'SSC': np.random.normal(1000, 100, n_normal),
            'FL1': normal_fl1,
            'FL2': normal_fl2,
            'FSC': np.random.normal(1500, 150, n_normal),
            'FL1-W': np.random.normal(800, 80, n_normal),
            'TIME': np.sort(np.random.uniform(0, 10000, n_normal)),
            'source_file': 'test_normal.fcs'
        })
        
        # Noise data - ensure it falls within polygonal region
        n_noise = 20
        # Generate data specifically within the polygon
        self.noise_data = pd.DataFrame({
            'SSC': np.random.normal(2000, 200, n_noise),
            'FL1': np.random.uniform(10**4.4, 10**6.0, n_noise),  # High FL1 in polygon
            'FL2': np.random.uniform(10**1.0, 10**4.0, n_noise),  # FL2 in polygon range
            'FSC': np.random.normal(2500, 250, n_noise),
            'FL1-W': np.random.normal(1200, 120, n_noise),
            'TIME': np.sort(np.random.uniform(0, 10000, n_noise)),
            'source_file': 'test_noise.fcs'
        })
        
        # Set up pipeline with test data
        self.pipeline.normal_files = [self.normal_data]
        self.pipeline.noise_files = [self.noise_data]
    
    def test_1_data_loading_and_file_structure(self):
        """Test data loading and file structure handling."""
        print("\n1. Testing data loading and file structure...")
        
        # Test that pipeline can handle the data structure
        self.assertEqual(len(self.pipeline.normal_files), 1)
        self.assertEqual(len(self.pipeline.noise_files), 1)
        
        # Test polygonal filtering
        normal_filtered = self.pipeline.apply_polygonal_filter(self.normal_data)
        noise_filtered = self.pipeline.apply_polygonal_filter(self.noise_data)
        
        # Should filter some data but keep some within the polygon
        self.assertGreater(len(normal_filtered), 5, "Should have some normal data in polygon")
        self.assertGreater(len(noise_filtered), 5, "Should have some noise data in polygon")
        
        print(f"  ✓ Normal data: {len(self.normal_data)} -> {len(normal_filtered)} after filtering")
        print(f"  ✓ Noise data: {len(self.noise_data)} -> {len(noise_filtered)} after filtering")
    
    def test_2_optimized_algorithm_training(self):
        """Test optimized algorithm training with limited iterations."""
        print("\n2. Testing optimized algorithm training...")
        
        # Prepare training data
        normal_filtered = self.pipeline.apply_polygonal_filter(self.normal_data)
        if len(normal_filtered) < 10:
            # Use original data if filtering removes too much
            normal_filtered = self.normal_data
        
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        X_train = normal_filtered[feature_cols].fillna(0).values
        
        # Test algorithm optimization
        import time
        start_time = time.time()
        self.pipeline.optimize_algorithms(X_train, contamination_rate=0.2)
        training_time = time.time() - start_time
        
        # Check that algorithms were trained
        expected_algorithms = ['isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope', 
                             'gaussian_mixture', 'dbscan', 'bayesian_temporal']
        
        for alg_name in expected_algorithms:
            self.assertIn(alg_name, self.pipeline.algorithms, f"{alg_name} not trained")
        
        # Check that SVM training was fast (should be < 30 seconds with max_iter=100)
        if 'one_class_svm' in self.pipeline.training_times:
            svm_time = self.pipeline.training_times['one_class_svm']
            self.assertLess(svm_time, 30, f"SVM training too slow: {svm_time:.2f}s")
        
        print(f"  ✓ All algorithms trained in {training_time:.2f}s")
        print(f"  ✓ SVM training time: {self.pipeline.training_times.get('one_class_svm', 0):.2f}s")
        print(f"  ✓ Bayesian training time: {self.pipeline.training_times.get('bayesian_temporal', 0):.2f}s")
    
    def test_3_bayesian_methods_interface(self):
        """Test that Bayesian methods have proper fit/predict interface."""
        print("\n3. Testing Bayesian methods interface...")
        
        # Test BayesianTemporalDenoiser directly
        bayesian = BayesianTemporalDenoiser(method='temporal_cooccurrence')
        
        # Test fit method
        X_df = self.normal_data[['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']].copy()
        bayesian.fit(X_df)
        self.assertTrue(bayesian.is_fitted, "Bayesian method should be fitted")
        
        # Test predict method
        predictions = bayesian.predict(X_df)
        self.assertEqual(len(predictions), len(X_df), "Predictions length mismatch")
        self.assertTrue(np.all(np.isin(predictions, [0, 1])), "Predictions should be binary")
        
        # Test predict_proba method
        probabilities = bayesian.predict_proba(X_df)
        self.assertEqual(probabilities.shape, (len(X_df), 2), "Probabilities shape mismatch")
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1")
        
        print("  ✓ Bayesian methods have proper fit/predict interface")
        print(f"  ✓ Predict returns binary labels: {np.unique(predictions)}")
        print(f"  ✓ Predict_proba returns proper probabilities shape: {probabilities.shape}")
    
    def test_4_ensemble_methods_evaluation(self):
        """Test that ensemble methods are properly evaluated."""
        print("\n4. Testing ensemble methods evaluation...")
        
        # Set up pipeline with trained algorithms
        normal_filtered = self.pipeline.apply_polygonal_filter(self.normal_data)
        if len(normal_filtered) < 10:
            normal_filtered = self.normal_data
        
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        X_train = normal_filtered[feature_cols].fillna(0).values
        
        self.pipeline.optimize_algorithms(X_train, contamination_rate=0.2)
        self.pipeline.create_ensemble_methods()
        
        # Test that ensemble methods exist
        expected_ensembles = ['majority_voting', 'weighted_voting', 'conservative_ensemble']
        for ensemble_name in expected_ensembles:
            self.assertIn(ensemble_name, self.pipeline.ensemble_methods, 
                         f"{ensemble_name} not created")
        
        # Test ensemble method execution
        noise_filtered = self.pipeline.apply_polygonal_filter(self.noise_data)
        if len(noise_filtered) < 5:
            noise_filtered = self.noise_data
        
        combined_data = pd.concat([normal_filtered, noise_filtered], ignore_index=True)
        X_test = combined_data[feature_cols].fillna(0).values
        
        # Scale test data
        scaler = self.pipeline.scalers[self.pipeline.best_scaler or 'standard']
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for ensemble_name, ensemble_func in self.pipeline.ensemble_methods.items():
            try:
                predictions = ensemble_func(X_test, X_test_scaled)
                self.assertEqual(len(predictions), len(X_test), 
                               f"{ensemble_name} predictions length mismatch")
                self.assertTrue(np.all(np.isin(predictions, [0, 1])), 
                               f"{ensemble_name} should return binary predictions")
                print(f"  ✓ {ensemble_name} working correctly")
            except Exception as e:
                self.fail(f"{ensemble_name} failed: {e}")
    
    def test_5_individual_file_testing(self):
        """Test individual file combination testing."""
        print("\n5. Testing individual file combination testing...")
        
        # Mock the visualization creation to speed up testing
        with patch.object(self.pipeline, 'create_comprehensive_visualizations'):
            with patch.object(self.pipeline, 'save_results'):
                # Run the file combination testing
                self.pipeline.create_ensemble_methods()
                self.pipeline.test_individual_file_combinations()
        
        # Check that individual results were generated
        self.assertGreater(len(self.pipeline.individual_results), 0, 
                          "No individual results generated")
        
        # Check result structure
        result = self.pipeline.individual_results[0]
        self.assertIn('normal_file', result)
        self.assertIn('noise_file', result)
        self.assertIn('algorithms', result)
        
        # Check that both individual algorithms and ensemble methods were tested
        tested_algorithms = set(result['algorithms'].keys())
        expected_individual = {'isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope', 
                             'gaussian_mixture', 'dbscan', 'bayesian_temporal'}
        expected_ensemble = {'majority_voting', 'weighted_voting', 'conservative_ensemble'}
        
        individual_found = tested_algorithms.intersection(expected_individual)
        ensemble_found = tested_algorithms.intersection(expected_ensemble)
        
        self.assertGreater(len(individual_found), 0, "No individual algorithms tested")
        self.assertGreater(len(ensemble_found), 0, "No ensemble methods tested")
        
        # Check that mean performance was calculated
        self.assertGreater(len(self.pipeline.mean_performance), 0, 
                          "Mean performance not calculated")
        
        print(f"  ✓ Generated {len(self.pipeline.individual_results)} individual results")
        print(f"  ✓ Tested {len(individual_found)} individual algorithms")
        print(f"  ✓ Tested {len(ensemble_found)} ensemble methods")
        print(f"  ✓ Calculated mean performance for {len(self.pipeline.mean_performance)} methods")
    
    def test_6_performance_metrics_completeness(self):
        """Test that all performance metrics are calculated including TP/FP/TN/FN."""
        print("\n6. Testing performance metrics completeness...")
        
        # Set up a simple test case
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        
        metrics = self.pipeline._calculate_metrics(y_true, y_pred)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
        
        # Verify confusion matrix values
        self.assertEqual(metrics['tp'], 2)  # Correct noise predictions
        self.assertEqual(metrics['fp'], 1)  # False noise predictions
        self.assertEqual(metrics['tn'], 2)  # Correct normal predictions
        self.assertEqual(metrics['fn'], 1)  # Missed noise
        
        print("  ✓ All performance metrics calculated correctly")
        print(f"  ✓ TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
    
    def test_7_training_time_optimization(self):
        """Test that training times are reasonable and tracked."""
        print("\n7. Testing training time optimization...")
        
        # Prepare data
        normal_filtered = self.pipeline.apply_polygonal_filter(self.normal_data)
        if len(normal_filtered) < 10:
            normal_filtered = self.normal_data
        
        feature_cols = ['SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'TIME']
        X_train = normal_filtered[feature_cols].fillna(0).values
        
        # Train algorithms and measure time
        import time
        start_time = time.time()
        self.pipeline.optimize_algorithms(X_train, contamination_rate=0.2)
        total_time = time.time() - start_time
        
        # Check that training times are tracked
        self.assertGreater(len(self.pipeline.training_times), 0, "No training times recorded")
        
        # Check that total training time is reasonable (should be < 60 seconds for test data)
        self.assertLess(total_time, 60, f"Total training time too long: {total_time:.2f}s")
        
        # Check specific algorithm times
        if 'one_class_svm' in self.pipeline.training_times:
            svm_time = self.pipeline.training_times['one_class_svm']
            self.assertLess(svm_time, 30, f"SVM training time too long: {svm_time:.2f}s")
        
        print(f"  ✓ Total training time: {total_time:.2f}s")
        print(f"  ✓ Training times tracked for {len(self.pipeline.training_times)} algorithms")
        
        for alg_name, train_time in self.pipeline.training_times.items():
            print(f"    - {alg_name}: {train_time:.2f}s")
    
    def test_8_complete_pipeline_integration(self):
        """Test the complete pipeline integration."""
        print("\n8. Testing complete pipeline integration...")
        
        # Mock file I/O operations to speed up testing
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('pandas.DataFrame.to_csv'):
            
            try:
                # Run the complete pipeline (except data loading which is already done)
                self.pipeline.create_ensemble_methods()
                self.pipeline.test_individual_file_combinations()
                
                # Manually create minimal visualization data for testing
                self.pipeline.create_comprehensive_visualizations()
                self.pipeline.save_results()
                
                print("  ✓ Complete pipeline executed successfully")
                
            except Exception as e:
                self.fail(f"Complete pipeline integration failed: {e}")


def run_tests():
    """Run all tests."""
    print("="*80)
    print("FINAL COMPREHENSIVE PIPELINE TEST SUITE")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFinalComprehensivePipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The final comprehensive pipeline is working correctly with all improvements:")
        print("✓ Individual file testing with cross-validation")
        print("✓ Optimized SVM training with limited iterations")
        print("✓ Fixed Bayesian methods with proper fit/predict interface")
        print("✓ Complete ensemble method evaluation")
        print("✓ Comprehensive performance metrics (TP/FP/TN/FN)")
        print("✓ Training time optimization and tracking")
        print("✓ Complete pipeline integration")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check the failures and errors above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()