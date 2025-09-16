# Flow Cytometry Denoising Pipeline

An advanced Bayesian denoising pipeline for flow cytometry data with multiple machine learning algorithms for noise detection and removal.

## Features

- **Multiple Bayesian Methods**: Temporal co-occurrence, Bayesian mixture models, Bayesian ridge regression, Dirichlet process mixtures, change point detection, and ensemble methods
- **Enhanced Feature Engineering**: Advanced temporal and statistical features for better noise discrimination
- **Ensemble Methods**: Weighted voting and conservative ensemble approaches
- **Comprehensive Evaluation**: Detailed TP/FP/TN/FN analysis per algorithm and file combination
- **Visualization**: Enhanced plots with smaller markers for better clarity

## Installation

### Option 1: Using pip (Recommended)

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install specific packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Option 2: Using Installation Scripts

**Linux/Mac:**
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Windows:**
```cmd
install_dependencies.bat
```

### Optional Dependencies

For advanced Bayesian Network methods, install:
```bash
pip install networkx pgmpy
```

## Usage

### Basic Usage

```python
from optimized_final_pipeline import OptimizedFlowCytometryPipeline

# Initialize and run the complete pipeline
pipeline = OptimizedFlowCytometryPipeline()
pipeline.run_complete_pipeline()
```

### Using Individual Bayesian Methods

```python
from improved_bayesian_denoising import BayesianTemporalDenoiser

# Available methods:
# - 'temporal_cooccurrence'
# - 'bayesian_mixture'
# - 'bayesian_ridge'
# - 'dirichlet_process'
# - 'change_point_detection'
# - 'ensemble_bayesian'

denoiser = BayesianTemporalDenoiser(method='ensemble_bayesian')
denoiser.fit(X_train)
predictions = denoiser.predict(X_test)
```

## Enhanced Bayesian Methods

1. **Temporal Co-occurrence**: Analyzes event patterns over time
2. **Bayesian Mixture**: Probabilistic clustering for noise identification
3. **Bayesian Ridge**: Reconstruction error-based anomaly detection
4. **Dirichlet Process**: Non-parametric Bayesian clustering
5. **Change Point Detection**: Identifies temporal anomalies
6. **Ensemble Bayesian**: Combines multiple methods with weighted voting

## Performance Improvements

The enhanced pipeline provides:
- Higher true positive rates through advanced temporal analysis
- Lower false positive rates via sophisticated feature engineering
- Better ensemble performance with method-specific weighting
- Robustness through multiple fallback mechanisms

## File Structure

```
├── improved_bayesian_denoising.py    # Enhanced Bayesian methods
├── optimized_final_pipeline.py       # Main pipeline
├── fcs_parser.py                     # FCS file parser
├── requirements.txt                  # Dependencies
├── install_dependencies.sh           # Linux/Mac installer
├── install_dependencies.bat          # Windows installer
├── data/                             # Flow cytometry data
│   ├── normal_files/                 # Normal samples
│   └── noise_files/                  # Noise samples
└── trained_models/                   # Saved models
```

## Output

The pipeline generates:
- Individual file performance metrics
- Mean performance across all combinations
- Enhanced visualizations with smaller markers
- Saved trained models for future use
- Comprehensive TP/FP/TN/FN analysis

## Requirements

- Python 3.8+
- Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib
- Optional: networkx, pgmpy for advanced Bayesian methods

## Testing

The project includes comprehensive test suites using pytest.

### Running Tests

#### Option 1: Full Test Suite (Recommended)
```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests with coverage
pytest

# Or use the test runner
python run_tests.py
```

#### Option 2: Basic Tests (No pytest required)
```bash
# Run basic functionality tests
python test_basic.py

# Or
python test_basic.py basic
```

#### Option 3: Quick Tests
```bash
# Run only fast tests
python run_tests.py quick
```

### Test Structure

```
tests/
├── test_bayesian_denoising.py    # Main Bayesian method tests
├── test_integration.py           # Integration and system tests
└── __init__.py

test_basic.py                     # Basic tests (no pytest required)
run_tests.py                      # Test runner script
pytest.ini                        # Pytest configuration
```

### Test Coverage

The test suite covers:
- ✅ Bayesian method initialization and configuration
- ✅ Feature extraction and temporal analysis
- ✅ Model fitting and prediction
- ✅ Performance metrics calculation
- ✅ Error handling and edge cases
- ✅ Synthetic data generation
- ✅ Integration testing

### Writing Tests

Add new tests to the `tests/` directory following the naming convention `test_*.py`.

Example test structure:
```python
import pytest
from improved_bayesian_denoising import BayesianTemporalDenoiser

def test_new_feature():
    """Test description."""
    # Test implementation
    assert True
```