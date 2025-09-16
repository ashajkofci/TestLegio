#!/usr/bin/env python3
"""
Simple tests for the Flow Cytometry Denoising Pipeline

This script performs basic validation tests to ensure the pipeline
components work correctly.
"""

import sys
import traceback
from fcs_parser import load_fcs_data, FCSParser
from flow_cytometry_pipeline import FlowCytometryPipeline


def test_fcs_parser():
    """Test the FCS parser functionality."""
    print("Testing FCS Parser...")
    
    try:
        # Test loading both files
        full_data = load_fcs_data('full_measurement.fcs')
        noise_data = load_fcs_data('only_noise.fcs')
        
        # Validate basic structure
        assert len(full_data) == 32875, f"Expected 32875 events, got {len(full_data)}"
        assert len(noise_data) == 41350, f"Expected 41350 events, got {len(noise_data)}"
        assert len(full_data.columns) == 6, f"Expected 6 parameters, got {len(full_data.columns)}"
        assert len(noise_data.columns) == 6, f"Expected 6 parameters, got {len(noise_data.columns)}"
        
        # Check required parameters
        expected_params = {'TIME', 'SSC', 'FL1', 'FL2', 'FSC', 'FL1-W'}
        full_params = set(full_data.columns)
        noise_params = set(noise_data.columns)
        
        assert expected_params == full_params, f"Parameter mismatch in full data: {expected_params} vs {full_params}"
        assert expected_params == noise_params, f"Parameter mismatch in noise data: {expected_params} vs {noise_params}"
        
        # Check data types
        for col in expected_params:
            assert full_data[col].dtype in ['int64', 'float64'], f"Invalid data type for {col}: {full_data[col].dtype}"
            assert noise_data[col].dtype in ['int64', 'float64'], f"Invalid data type for {col}: {noise_data[col].dtype}"
        
        print("✓ FCS Parser tests passed")
        return True
        
    except Exception as e:
        print(f"✗ FCS Parser test failed: {e}")
        traceback.print_exc()
        return False


def test_pipeline_initialization():
    """Test pipeline initialization and data loading."""
    print("Testing Pipeline Initialization...")
    
    try:
        pipeline = FlowCytometryPipeline()
        pipeline.load_data('full_measurement.fcs', 'only_noise.fcs')
        
        # Check combined data
        assert pipeline.combined_data is not None, "Combined data is None"
        assert len(pipeline.combined_data) == 74225, f"Expected 74225 combined events, got {len(pipeline.combined_data)}"
        assert 'source' in pipeline.combined_data.columns, "Source column missing"
        assert 'original_index' in pipeline.combined_data.columns, "Original index column missing"
        
        # Check source distribution
        source_counts = pipeline.combined_data['source'].value_counts()
        assert source_counts['full_measurement'] == 32875, f"Wrong full measurement count: {source_counts['full_measurement']}"
        assert source_counts['noise_only'] == 41350, f"Wrong noise count: {source_counts['noise_only']}"
        
        print("✓ Pipeline initialization tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization test failed: {e}")
        traceback.print_exc()
        return False


def test_fl1_filtering():
    """Test FL1 threshold filtering."""
    print("Testing FL1 Filtering...")
    
    try:
        pipeline = FlowCytometryPipeline()
        pipeline.load_data('full_measurement.fcs', 'only_noise.fcs')
        pipeline.apply_fl1_threshold()
        
        # Check filtering results
        assert pipeline.filtered_data is not None, "Filtered data is None"
        assert len(pipeline.filtered_data) > 0, "No data remaining after filtering"
        assert all(pipeline.filtered_data['FL1'] > 2e4), "Some data below FL1 threshold remains"
        
        # Check that filtering maintains required columns
        required_cols = {'TIME', 'SSC', 'FL1', 'FL2', 'FSC', 'FL1-W', 'source', 'original_index'}
        filtered_cols = set(pipeline.filtered_data.columns)
        assert required_cols.issubset(filtered_cols), f"Missing columns after filtering: {required_cols - filtered_cols}"
        
        print("✓ FL1 filtering tests passed")
        return True
        
    except Exception as e:
        print(f"✗ FL1 filtering test failed: {e}")
        traceback.print_exc()
        return False


def test_noise_detection():
    """Test noise detection algorithms."""
    print("Testing Noise Detection...")
    
    try:
        pipeline = FlowCytometryPipeline()
        pipeline.load_data('full_measurement.fcs', 'only_noise.fcs')
        pipeline.apply_fl1_threshold()
        
        # Run noise detection
        results = pipeline.detect_noise_patterns_advanced()
        
        # Check results structure
        expected_methods = {'isolation_forest_tuned', 'local_outlier_factor_tuned', 'dbscan_tuned', 'one_class_svm', 'elliptic_envelope', 'gaussian_mixture', 'ensemble_advanced'}
        result_methods = set(results.keys())
        # Allow subset since we may have different algorithms
        common_methods = expected_methods.intersection(result_methods)
        assert len(common_methods) >= 3, f"Not enough common methods found: {common_methods}"
        
        # Check accuracy values are reasonable
        for method, accuracy in results.items():
            assert 0 <= accuracy <= 1, f"Invalid accuracy for {method}: {accuracy}"
        
        # Check that detection columns were added
        detection_cols = [col for col in pipeline.filtered_data.columns if any(x in col for x in ['tuned', 'svm', 'envelope', 'mixture', 'ensemble'])]
        assert len(detection_cols) >= 3, f"Not enough detection columns found: {detection_cols}"
        
        print("✓ Noise detection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Noise detection test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("FLOW CYTOMETRY PIPELINE TESTS")
    print("=" * 50)
    
    tests = [
        test_fcs_parser,
        test_pipeline_initialization,
        test_fl1_filtering,
        test_noise_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)