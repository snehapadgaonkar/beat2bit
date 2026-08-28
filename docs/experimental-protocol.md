# Beat2Bit Experimental Protocol

## 1. Overview
This document outlines the standardized experimental procedures for ECG arrhythmia detection research in the Beat2Bit project. Following this protocol ensures reproducibility, validity, and comparability of results across different experiments and optimizations.

## 2. Experimental Design Principles

### 2.1. Hypothesis-Driven Research
All experiments must be designed around a clear, testable hypothesis related to:
- Model architecture modifications
- Optimization techniques (pruning, quantization)
- Training procedure variations
- Preprocessing changes

### 2.2. Controlled Variables
To ensure valid comparisons, the following variables must be held constant unless explicitly being tested:
- Dataset version and preprocessing pipeline
- Random seeds for reproducibility
- Evaluation metrics and statistical tests
- Hardware specifications for latency measurements (simulated)

### 2.3. Replication Requirements
Each experimental condition must be replicated sufficient times to enable statistical analysis:
- Minimum of 5 independent runs for preliminary experiments
- Minimum of 10 independent runs for publication-quality results
- Cross-validation folds where applicable

## 3. Standard Experimental Workflow

### 3.1. Experiment Initialization
1. **Define Research Question**: Clearly state the hypothesis being tested
2. **Select Baseline**: Choose appropriate baseline model for comparison
3. **Design Experiment**: Specify independent and dependent variables
4. **Create Configuration**: Use YAML/JSON to document all parameters
5. **Initialize Tracking**: Register experiment in the tracking system

### 3.2. Data Preparation
1. **Dataset Selection**: Use only openly available ECG datasets
2. **Version Control**: Document exact dataset versions used
3. **Preprocessing**: Apply standardized preprocessing pipeline
4. **Splitting**: Use patient-aware splits to prevent data leakage
5. **Validation**: Verify processed data integrity

### 3.3. Model Development
1. **Architecture Definition**: Specify model layers, parameters, and connections
2. **Implementation**: Implement using standardized model factory functions
3. **Verification**: Confirm architecture matches specification
4. **Parameter Counting**: Document total and trainable parameters

### 3.4. Training Procedure
1. **Configuration**: Document all training hyperparameters
2. **Random Seeds**: Set and record all random seeds
3. **Training Execution**: Run training with monitoring
4. **Checkpointing**: Save model checkpoints at regular intervals
5. **Early Stopping**: Apply if configured to prevent overfitting

### 3.5. Optimization Application
1. **Pruning** (if applicable):
   - Specify pruning method and schedule
   - Document target sparsity level
   - Apply fine-tuning procedure
   - Verify sparsity achievement
2. **Quantization** (if applicable):
   - Select quantization method (post-training, quantization-aware)
   - Specify precision (int8, int16, float16)
   - Use representative dataset for calibration
   - Validate numerical equivalence

### 3.6. Evaluation Protocol
1. **Metric Calculation**: Compute all standard metrics
2. **Statistical Testing**: Apply appropriate hypothesis tests
3. **Confidence Intervals**: Calculate bootstrap confidence intervals
4. **Comparison Testing**: Compare against baseline and related work
5. **Robustness Testing**: Evaluate performance under noise conditions

### 3.7. Documentation and Reporting
1. **Results Logging**: Save all metrics, configurations, and artifacts
2. **Visualization**: Generate standard plots and tables
3. **Report Generation**: Create comprehensive markdown report
4. **Archiving**: Store all experiment artifacts in tracking system

## 4. Standardized Evaluation Metrics

### 4.1. Primary Metrics (AAMI EC57 Compliant)
- **Sensitivity (Se)**: TP / (TP + FN)
- **Positive Predictivity (+P)**: TP / (TP + FP)
- **Effectiveness (E)**: √(Se × +P)

### 4.2. Standard Classification Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall/Sensitivity**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity**: TN / (TN + FP)

### 4.3. Complexity Metrics
- **Parameter Count**: Total number of model parameters
- **Model Size**: Memory footprint in MB (FP32 and INT8)
- **Computational Complexity**: FLOPs and GFLOPs
- **Inference Latency**: Mean and percentile latencies (ms)

### 4.4. Statistical Metrics
- **Confidence Intervals**: 95% bootstrap CIs for all metrics
- **Effect Sizes**: Cohen's d for continuous metrics
- **P-values**: From appropriate hypothesis tests
- **Power Analysis**: Post-hoc power calculations

## 5. Dataset Standards

### 5.1. Primary Dataset: MIT-BIH Arrhythmia Database
- **Source**: PhysioNet (https://physionet.org/content/mitdb/1.0.0/)
- **Version**: Use the official release as of experiment date
- **Preprocessing**: Apply standardized AAMI patient-aware split
- **Annotation Mapping**: 
  - Normal: ['N', 'L', 'R', 'e', 'j']
  - Abnormal: ['V', 'E', 'A', 'a', 'J', 'S', 'F']
- **Window Size**: 180 samples (90 pre-R-peak, 90 post-R-peak)
- **Sampling Rate**: 360 Hz (standard for MIT-BIH)

### 5.2. Secondary Open-Source Datasets
When incorporating additional datasets for validation:
- **St. Petersburg INCART Database**: PhysioNet, free for research
- **PTB Diagnostic ECG Database**: PhysioNet, free for research
- **Long-Term AF Database**: PhysioNet, free for research
- **Requirement**: Only use datasets with permissive research licenses

### 5.3. Data Versioning
- Document exact dataset versions used
- Record download dates and sources
- Note any preprocessing modifications
- Maintain hash/checksums for data integrity verification

## 6. Reproducibility Requirements

### 6.1. Random Seed Management
- Set numpy random seed: `np.random.seed(seed_value)`
- Set TensorFlow random seed: `tf.random.set_seed(seed_value)`
- Set Python random seed: `random.seed(seed_value)`
- Record all seeds in experiment configuration
- Use same seeds for comparative experiments

### 6.2. Environment Specification
- Record exact dependency versions (use `environment.yml`)
- Document Python version and system information
- Specify hardware used for latency measurements (if applicable)
- Note any non-deterministic operations and their handling

### 6.3. Experiment Tracking
- Use the experiment tracking system to log:
  - Complete configuration
  - Git commit hash at experiment time
  - Start and end timestamps
  - All intermediate and final results
  - Generated artifacts (models, plots, logs)

### 6.4. Protocol Adherence Verification
- Include protocol version in all experiment documentation
- Verify that all steps were followed as specified
- Document any deviations and their justification
- Enable independent replication by other researchers

## 7. Statistical Analysis Guidelines

### 7.1. Hypothesis Testing
- **Normality Testing**: Verify assumptions before parametric tests
- **Paired vs Unpaired**: Choose appropriate test based on experimental design
- **Multiple Comparisons**: Apply corrections when testing multiple hypotheses
- **Effect Size Reporting**: Always report effect sizes alongside p-values
- **Confidence Intervals**: Prefer confidence intervals over p-values alone

### 7.2. Comparison Methods
- **Baseline Comparison**: Compare all variants against common baseline
- **Statistical Significance**: Use α = 0.05 unless justified otherwise
- **Practical Significance**: Consider minimum clinically important difference
- **Non-inferiority Testing**: When appropriate for optimization studies

### 7.3. Reporting Standards
- **Descriptive Statistics**: Report mean, median, std, CI
- **Inferential Statistics**: Report test statistic, df, p-value, effect size
- **Visualization**: Use appropriate plots (box plots, bar charts, etc.)
- **Tables**: Follow journal standards for statistical reporting

## 8. Quality Assurance Checklist

### 8.1. Pre-Experiment Verification
[ ] Hypothesis clearly defined and testable
[ ] Experimental design appropriate for hypothesis
[ ] Configuration file created and validated
[ ] Experiment registered in tracking system
[ ] Random seeds documented and set
[ ] Environment specifications recorded
[ ] Dataset versions verified and documented

### 8.2. During Experiment
[ ] Training progress monitored and logged
[ ] Checkpointing enabled and verified
[ ] Optimization procedures followed as specified
[ ] Intermediate results saved periodically
[ ] Any deviations documented in real-time

### 8.3. Post-Experiment Verification
[ ] All results saved and backed up
[ ] Statistical tests performed and interpreted correctly
[ ] Comparison with baseline and literature completed
[ ] Comprehensive report generated
[ ] Experiment marked as completed in tracking system
[ ] Reproducibility package assembled (config, code, data specs)

### 8.4. Documentation Completeness
[ ] All methods described in sufficient detail for replication
[ ] Results presented clearly with appropriate uncertainty
[ ] Limitations and potential biases acknowledged
[ ] Future work directions identified
[ ] References to related work properly cited

## 9. Safety and Ethics Considerations

### 9.1. Data Privacy
- All used datasets are openly available and de-identified
- No patient health information (PHI) is processed
- Compliance with data use agreements for PhysioNet datasets

### 9.2. Clinical Relevance
- Research focuses on algorithmic improvements
- Any clinical claims must be supported by appropriate validation
- Limitations of algorithmic approach clearly stated
- No claims of diagnostic capability without clinical validation

### 9.3. Responsible Research
- Transparent reporting of both positive and negative results
- Proper attribution of methods and ideas from prior work
- Open sharing of code, configurations, and protocols
- Commitment to scientific rigor and reproducibility

## 10. References
1. ANSI/AAMI/ISO 5720-2012: Electrocardiography - Monitoring - Performance of single lead and two lead systems
2. AAMI EC57:2012/European Standard EN 1060-1:2017: Ambulatory electrocardiographs - Safety and performance
3. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
4. Kempfner, J., et al. (2020). Reproducibility in biomedical research: Openness and transparency. Nature Medicine, 26, 183-186.

---
*Protocol Version: 1.0.0*
*Last Updated: $(date)*
*Compatible with Beat2Bit Benchmarking Suite v1.0.0*