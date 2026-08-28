# Beat2Bit Reproducibility Guide

## 1. Introduction
This guide provides step-by-step instructions for reproducing experiments and results in the Beat2Bit project. Following these procedures ensures that other researchers can verify and build upon the work presented.

## 2. Getting Started

### 2.1. System Requirements
- **Operating System**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4 GB RAM (8 GB recommended)
- **Storage**: Minimum 2 GB free space for datasets and outputs
- **Dependencies**: Listed in `environment.yml`

### 2.2. Quick Start
```bash
# Clone the repository
git clone https://github.com/snehapadgaonkar/beat2bit.git
cd beat2bit
git checkout arena/01a03cc6-beat2bit

# Set up environment
conda env create -f environment.yml
conda activate beat2bit

# Run the baseline experiment
python -m src.experiments.tracker create_experiment \
    --name "baseline_reproduction" \
    --config configs/baseline_config.yaml \
    --description "Reproduction of baseline ECG arrhythmia detection experiment"
```

## 3. Environment Setup

### 3.1. Using Conda (Recommended)
The project provides an `environment.yml` file for exact environment reproduction:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate beat2bit

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### 3.2. Using pip (Alternative)
If you prefer pip, use the requirements file:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development/testing dependencies

# Install TensorFlow separately (version-specific)
pip install tensorflow==2.13.0  # Check environment.yml for exact version
```

### 3.3. Environment Verification
Run the environment verification script:

```bash
python scripts/verify_environment.py
```

This script checks:
- Python version
- Critical package versions
- GPU availability (if applicable)
- Disk space
- Memory availability

## 4. Data Acquisition

### 4.1. MIT-BIH Arrhythmia Database
The primary dataset used is the MIT-BIH Arrhythmia Database, available from PhysioNet:

```bash
# Using wfdb toolkit (installed via dependencies)
python -c "
import wfdb
wfdb.dl_database('mitdb', './data/mitdb')
"

# Alternative: Manual download
# Visit: https://physionet.org/content/mitdb/1.0.0/
# Download all .dat and .hea files for records 100-124, 200-234
```

### 4.2. Dataset Verification
Verify downloaded data integrity:

```bash
python scripts/verify_dataset.py --dataset mitdb
```

This script checks:
- File completeness
- Header file validity
- Signal data readability
- Annotation file compatibility

### 4.3. Dataset Versioning
Record the exact dataset version used:
- MIT-BIH Arrhythmia Database version: 1.0.0
- PhysioNet release date: [Check current release]
- Download date: [Record when you downloaded]
- File checksums: [Store SHA256 hashes]

## 5. Running Experiments

### 5.1. Using the Experiment Tracker
The recommended way to run experiments is through the experiment tracking system:

```bash
# List available configurations
ls configs/

# Create and run an experiment
python -m src.experiments.tracker create_experiment \
    --name "my_experiment" \
    --config configs/my_config.yaml \
    --description "Description of my experiment"

# The tracker will automatically:
# 1. Create experiment directory
# 2. Save configuration
# 3. Log git information
# 4. Set up logging
# 5. Mark experiment as "created"
```

### 5.2. Manual Experiment Execution
For more control, you can run experiments manually:

```bash
# Activate environment
conda activate beat2bit

# Set random seeds for reproducibility (important!)
export PYTHONHASHSEED=0
python -c "
import random
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"

# Run the training script
python -m src.training.train --full --epochs 20

# Run evaluation
python -m src.evaluation.evaluate --model models/saved/baseline.h5

# Run benchmarking
python -m src.benchmarking.benchmark_runner --model models/saved/baseline.h5
```

### 5.3. Configuration-Driven Experiments
All experiments should be driven by configuration files:

```yaml
# Example: configs/baseline_config.yaml
experiment_name: baseline_ecg_detection
description: Baseline 1D CNN for ECG arrhythmia detection
version: "1.0.0"

# Data configuration
dataset_name: MIT-BIH
dataset_version: "1.0.0"
data_split_strategy: patient_aware_aami
test_size: 0.2
validation_size: 0.1
random_seed: 42
window_size: 180
sampling_rate: 360
normalization_method: zscore

# Model configuration
model_architecture: baseline_1d_cnn
input_shape: [180, 1]
num_classes: 1

# Training configuration
epochs: 10
batch_size: 128
optimizer: adam
learning_rate: 0.001
validation_split: 0.1
class_weight_method: balanced

# Optimization configuration
apply_pruning: true
pruning_sparsity: 0.7
pruning_schedule: polynomial_decay
apply_quantization: true
quantization_type: int8

# Evaluation configuration
metrics_to_report:
  - accuracy
  - precision
  - recall
  - f1_score
  - ami_sensitivity
  - ami_positive_predictivity
  - ami_effectiveness
confidence_level: 0.95
bootstrap_samples: 1000
```

### 5.4. Running Specific Experiment Types

#### Baseline Experiment
```bash
python -m src.experiments.tracker create_experiment \
    --name "baseline_baseline" \
    --config configs/baseline.yaml \
    --description "Baseline FP32 model without optimizations"
```

#### Pruning Experiment
```bash
python -m src.experiments.tracker create_experiment \
    --name "pruning_experiment" \
    --config configs/pruning_70.yaml \
    --description "Model with 70% magnitude pruning"
```

#### Quantization Experiment
```bash
python -m src.experiments.tracker create_experiment \
    --name "quantization_experiment" \
    --config configs/quantization_int8.yaml \
    --description "Model with INT8 post-training quantization"
```

#### Combined Optimization Experiment
```bash
python -m src.experiments.tracker create_experiment \
    --name "combined_optimization" \
    --config configs/pruning_quantization.yaml \
    --description "Model with 70% pruning + INT8 quantization"
```

## 6. Verifying Results

### 6.1. Checking Experiment Status
```bash
# List all experiments
python -m src.experiments.tracker list_experiments

# List only completed experiments
python -m src.experiments.tracker list_experiments --status completed

# Get details for specific experiment
python -m src.experiments.tracker get_experiment --exp_id <experiment_id>
```

### 6.2. Viewing Results
Experiment results are stored in the experiment directory:
```
experiments/
├── <experiment_id>/
│   ├── experiment.json          # Full experiment record
│   ├── config/
│   │   └── experiment.yaml      # Used configuration
│   ├── results/
│   │   ├── metrics.json         # Final metrics
│   │   ├── evaluation_report.md # Detailed evaluation report
│   │   └── benchmark_report.md  # Benchmarking results
│   ├── models/
│   │   ├── baseline.h5          # Keras model
│   │   └── model.tflite         # TFLite model (if quantized)
│   ├── logs/
│   │   └── experiment.log       # Execution log
│   └── plots/
│       ├── training_curves.png  # Training loss/accuracy
│       └── confusion_matrix.png # Confusion matrix plot
```

### 6.3. Reproducing Specific Results
To reproduce exact results from a published experiment:

```bash
# 1. Get the exact configuration used
python -m src.experiments.tracker get_config --exp_id <published_experiment_id>

# 2. Save it to a file
python -m src.experiments.tracker get_config --exp_id <published_experiment_id> --output configs/reproduction.yaml

# 3. Create new experiment with same configuration
python -m src.experiments.tracker create_experiment \
    --name "reproduction_attempt" \
    --config configs/reproduction.yaml \
    --description "Attempt to reproduce results from <published_experiment_id>"

# 4. Ensure same random seeds are used
# 5. Run and compare results
```

### 6.4. Comparing Experiments
Compare results between experiments:

```bash
python -m src.experiments.tracker compare_experiments \
    --exp_id <exp_id_1> <exp_id_2> \
    --metrics accuracy f1_score ami_sensitivity \
    --statistical-test paired_ttest
```

## 7. Troubleshooting

### 7.1. Common Issues

#### Environment Issues
- **TensorFlow not found**: Ensure you activated the correct environment
- **Version mismatch**: Check that versions in `environment.yml` match what's installed
- **GPU/CPU mismatch**: Some operations may behave differently on GPU vs CPU

#### Data Issues
- **Dataset not found**: Verify data download completed successfully
- **File permissions**: Ensure you have read access to data files
- **Corrupted files**: Try re-downloading if verification fails

#### Reproducibility Issues
- **Different results**: Check that all random seeds are set identically
- **Non-deterministic operations**: Some TensorFlow operations may have slight variations
- **Version drift**: Ensure you're using the exact same code version (check git commit)

### 7.2. Debugging Tips
1. **Check the logs**: Look in `experiments/<exp_id>/logs/experiment.log`
2. **Verify configuration**: Compare used configuration with intended one
3. **Check intermediate results**: Look for checkpoint files or intermediate outputs
4. **Run with verbose output**: Increase logging level for more details
5. **Isolate components**: Test data loading, model creation, and evaluation separately

### 7.3. Getting Help
- Check the project's issue tracker
- Review the documentation in `docs/`
- Look at example configurations in `configs/`
- Examination of existing experiment directories

## 8. Best Practices for Reproducible Research

### 8.1. Before Starting
- [ ] Define clear research question/hypothesis
- [ ] Select appropriate baseline for comparison
- [ ] Plan experimental design and controls
- [ ] Prepare configuration files
- [ ] Ensure environment is properly set up

### 8.2. During Experimentation
- [ ] Set and record all random seeds
- [ ] Use version control for code (commit frequently)
- [ ] Log all parameters and intermediate results
- [ ] Backup configuration and data references
- [ ] Document any deviations from plan

### 8.3. After Completion
- [ ] Verify all results are saved and accessible
- [ ] Generate comprehensive reports
- [ ] Ensure experiment is marked as completed
- [ ] Share configuration and results (if appropriate)
- [ ] Prepare reproducibility package

### 8.4. For Publication
- [ ] Include environment specification (environment.yml)
- [ ] Provide exact code version (git commit hash)
- [ ] Share all configuration files used
- [ ] Document random seeds and preprocessing parameters
- [ ] Provide instructions for data acquisition
- [ ] Include statistical details and effect sizes

## 9. Reproducibility Checklist

### 9.1. Minimum Requirements for Reproduction
[ ] Code repository access (with specific commit/tag)
[ ] Environment specification (environment.yml or requirements.txt)
[ ] Configuration files used for experiments
[ ] Instructions for data acquisition
[ ] Random seeds used
[ ] Description of computational environment

### 9.2. Ideal Reproducibility Package
[ ] All of the above, plus:
[ ] Pre-processed data (if transformation is complex)
[ ] Trained model checkpoints
[ ] Evaluation scripts and expected outputs
[ ] Visualization generation scripts
[ ] Detailed methodology documentation
[ ] Negative/null results and failed attempts

### 9.3. Verification Steps
[ ] Code runs without errors in fresh environment
[ ] Data downloads and processes correctly
[ ] Models train and produce expected outputs
[ ] Evaluation metrics match reported values (within statistical uncertainty)
[ ] Optimization techniques produce reported effects
[ ] Statistical tests yield similar conclusions

## 10. Example Reproduction Workflow

Here's a complete example of reproducing a baseline experiment:

```bash
# 1. Obtain the code at specific version
git clone https://github.com/snehapadgaonkar/beat2bit.git
cd beat2bit
git checkout <specific_commit_hash>

# 2. Set up environment
conda env create -f environment.yml
conda activate beat2bit

# 3. Verify environment
python scripts/verify_environment.py

# 4. Acquire data (if not already present)
python -c "import wfdb; wfdb.dl_database('mitdb', './data/mitdb')"
python scripts/verify_dataset.py --dataset mitdb

# 5. Create experiment using published configuration
# (Assume we have the configuration from the original experiment)
python -m src.experiments.tracker create_experiment \
    --name "baseline_reproduction" \
    --config configs/published_baseline.yaml \
    --description "Reproduction of baseline experiment from paper"

# 6. Run the experiment (if not auto-run by tracker)
# The tracker should have marked it as "created", now we run it:
python -m src.experiments.tracker run_experiment --exp_id <generated_exp_id>

# 7. Alternatively, run manually with same configuration
export PYTHONHASHSEED=0
python -c "
import random
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"
python -m src.training.train --full --config configs/published_baseline.yaml

# 8. Verify results
python -m src.experiments.tracker get_experiment --exp_id <exp_id>
# Check that metrics match published values within expected uncertainty

# 9. Compare with baseline if applicable
python -m src.experiments.tracker compare_experiments \
    --exp_id <reproduction_exp_id> <baseline_exp_id> \
    --metrics accuracy f1_score
```

## 11. Contact and Support
If you encounter difficulties reproducing results:
1. Check the project's issue tracker for similar problems
2. Review the documentation thoroughly
3. Ensure you're using the exact versions specified
4. Consider reaching out to the authors with specific details about:
   - What you tried to reproduce
   - What steps you followed
   - What exactly failed or differed
   - Error messages or unexpected results
   - Your environment specifications

---
*Reproducibility Guide Version: 1.0.0*
*Last Updated: $(date)*
*Compatible with Beat2Bit Experimental Protocol v1.0.0*