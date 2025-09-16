#!/usr/bin/env python3
"""
Test suite for the enhanced flow cytometry pipeline
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_pipeline import EnhancedFlowCytometryPipeline
    from fcs_parser import load_fcs_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are present")
    sys.exit(1)


class TestEnhancedPipeline(unittest.TestCase):
    """Test cases for enhanced flow cytometry pipeline."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.pipeline = EnhancedFlowCytometryPipeline()
        
    def test_1_data_loading(self):
        """Test 1: Data loading functionality."""
        print("Test 1: Testing data loading...")
        
        # Test data loading
        self.pipeline.load_data()
        
        # Verify data was loaded
        self.assertIsNotNone(self.pipeline.normal_data, "Normal data should be loaded")
        self.assertIsNotNone(self.pipeline.noise_data, "Noise data should be loaded")
        
        # Check data shapes
        self.assertGreater(len(self.pipeline.normal_data), 0, "Normal data should not be empty")
        self.assertGreater(len(self.pipeline.noise_data), 0, "Noise data should not be empty")
        
        # Check required columns exist
        required_cols = ['TIME', 'SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'source']
        for col in required_cols:
            self.assertIn(col, self.pipeline.normal_data.columns, f"Normal data missing column: {col}")
            self.assertIn(col, self.pipeline.noise_data.columns, f"Noise data missing column: {col}")
        
        # Check source labels are correct
        self.assertTrue(all(self.pipeline.normal_data['source'] == 'normal'), "Normal data should have 'normal' source")
        self.assertTrue(all(self.pipeline.noise_data['source'] == 'noise'), "Noise data should have 'noise' source")
        
        print("✓ Test 1 passed: Data loading works correctly")
        
    def test_2_polygonal_filtering(self):
        """Test 2: Polygonal filtering functionality."""
        print("Test 2: Testing polygonal filtering...")
        
        # Load data first
        self.pipeline.load_data()
        
        # Apply polygonal filter
        self.pipeline.apply_polygonal_filter()
        
        # Verify filtered data exists
        self.assertIsNotNone(self.pipeline.filtered_normal, "Filtered normal data should exist")
        self.assertIsNotNone(self.pipeline.filtered_noise, "Filtered noise data should exist")
        
        # Check that filtering reduced data size
        original_normal = len(self.pipeline.normal_data)
        filtered_normal = len(self.pipeline.filtered_normal)
        self.assertLessEqual(filtered_normal, original_normal, "Filtering should reduce or maintain normal data size")
        
        original_noise = len(self.pipeline.noise_data)
        filtered_noise = len(self.pipeline.filtered_noise)
        self.assertLessEqual(filtered_noise, original_noise, "Filtering should reduce or maintain noise data size")
        
        print(f"✓ Test 2 passed: Polygonal filtering works correctly")
        
    def test_3_algorithm_training(self):
        """Test 3: Algorithm training functionality."""
        print("Test 3: Testing algorithm training...")
        
        # Load and filter data
        self.pipeline.load_data()
        self.pipeline.apply_polygonal_filter()
        
        # Train algorithms
        self.pipeline.train_algorithms()
        
        # Check that models were trained
        self.assertGreater(len(self.pipeline.trained_models), 0, "Some models should be trained")
        
        print(f"✓ Test 3 passed: Algorithm training works correctly")
        
    def test_4_noise_testing(self):
        """Test 4: Testing on noise data functionality."""
        print("Test 4: Testing noise data evaluation...")
        
        # Load, filter, and train
        self.pipeline.load_data()
        self.pipeline.apply_polygonal_filter()
        self.pipeline.train_algorithms()
        
        # Test on noise data
        if len(self.pipeline.filtered_noise) > 0:
            noise_results = self.pipeline.test_on_noise_data()
            self.assertIsNotNone(noise_results, "Noise testing results should be generated")
        
        print("✓ Test 4 passed: Noise testing works correctly")


def run_tests():
    """Run all tests and provide summary."""
    print("="*70)
    print("ENHANCED FLOW CYTOMETRY PIPELINE - TEST SUITE")
    print("="*70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedPipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All tests passed! Enhanced pipeline is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
    
    print("="*70)
    return success


if __name__ == "__main__":
    success = run_tests()