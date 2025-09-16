#!/usr/bin/env python3
"""
Test runner script for the Bayesian denoising pipeline

This script provides convenient ways to run the test suite.
"""

import subprocess
import sys
import os

def run_tests():
    """Run the complete test suite."""
    print("Running Bayesian Denoising Test Suite...")
    print("=" * 50)

    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=improved_bayesian_denoising",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except FileNotFoundError:
        print("pytest not found. Please install pytest:")
        print("pip install pytest pytest-cov")
        return False

def run_quick_tests():
    """Run only the fastest tests."""
    print("Running Quick Tests...")
    print("=" * 30)

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-k", "not slow"
    ]

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except FileNotFoundError:
        print("pytest not found. Please install pytest:")
        print("pip install pytest")
        return False

def run_specific_test(test_name):
    """Run a specific test."""
    print(f"Running Test: {test_name}")
    print("=" * 30)

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/test_{test_name}.py",
        "-v",
        "--tb=short"
    ]

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except FileNotFoundError:
        print("pytest not found. Please install pytest:")
        print("pip install pytest")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            success = run_quick_tests()
        elif sys.argv[1] == "specific" and len(sys.argv) > 2:
            success = run_specific_test(sys.argv[2])
        else:
            print("Usage: python run_tests.py [quick|specific <test_name>]")
            success = False
    else:
        success = run_tests()

    sys.exit(0 if success else 1)