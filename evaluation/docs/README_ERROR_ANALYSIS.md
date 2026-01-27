# Comprehensive Error Analysis

This directory contains scripts for analyzing medical error detection performance across different medical specialties, error types, and demographic groups.

## Overview

The error analysis system provides:

- **Overall Performance Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Medical Specialty Analysis**: Performance breakdown by medical specialty (Cardiovascular, Respiratory, Infectious Disease, etc.)
- **Demographic Analysis**: Performance by age groups and gender
- **Cross-Demographic Analysis**: Combined age × gender performance heatmaps
- **Error Type Analysis**: Detailed breakdown by medical error categories
- **Comprehensive Visualizations**: 5 high-quality plot files with multiple subplots

## Files

- `error_analysis_with_plots.py` - Main analysis script with visualization generation
- `demographic_analysis.py` - Original demographic analysis script (text-based)
- `run_error_analysis.sh` - Local execution script
- `run_error_analysis.slurm` - SLURM batch job script for cluster execution

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### Local Execution

```bash
# Basic usage (uses default paths)
python evaluation/error_analysis_with_plots.py

# With custom paths
python evaluation/error_analysis_with_plots.py \
    --validation-file test_data/validation.json \
    --results-dir logs/debates \
    --output-dir evaluation/plots

# Or use the bash script
bash evaluation/run_error_analysis.sh
```

### SLURM Cluster Execution

```bash
# Submit to SLURM scheduler
sbatch evaluation/run_error_analysis.slurm

# With custom configuration
VALIDATION_FILE=test_data/validation.json \
RESULTS_DIR=logs/debates \
OUTPUT_DIR=evaluation/plots \
sbatch evaluation/run_error_analysis.slurm

# Check job status
squeue -u $USER

# View output logs
tail -f evaluation/logs/error_analysis_*.out
```

### Environment Variables

You can override default paths using environment variables:

```bash
export VALIDATION_FILE="path/to/validation.json"
export RESULTS_DIR="path/to/results"
export OUTPUT_DIR="path/to/output"
```

## Input Data Format

### Validation File (JSON)
```json
[
  {
    "id": "case-001",
    "text": "Medical note text...",
    "label": 1,
    "error_type": "causalOrganism",
    "split": "test"
  }
]
```

### Result Files (JSON)
Location: `logs/debates/result_*.json` or `logs/debates/debate_*.json`

```json
{
  "case_id": "case-001",
  "ground_truth": 1,
  "predicted_label": 1,
  "final_answer": "INCORRECT",
  "confidence_score": 8,
  "confidence_normalized": 0.8,
  "error_type": "causalOrganism",
  "winner": "Expert A",
  "execution_time": 15.3
}
```

## Output Files

The analysis generates the following files in the output directory:

### Visualization Plots (PNG files)

1. **overall_performance.png**
   - Confusion matrix
   - Accuracy by ground truth label
   - Performance metrics (Accuracy, Precision, Recall, F1)
   - Prediction distribution pie chart

2. **specialty_analysis.png**
   - Accuracy by medical specialty
   - Case distribution by specialty
   - Error types by specialty (stacked bar)
   - Specialty performance heatmap

3. **demographic_analysis.png**
   - Accuracy by age group
   - Accuracy by gender
   - Age distribution histogram
   - Error types by age group
   - Error types by gender
   - Gender distribution by age group

4. **cross_demographic_analysis.png**
   - Accuracy heatmap: Age Group × Gender
   - Case count heatmap: Age Group × Gender
   - Grouped bar chart: Accuracy by age and gender
   - Specialty accuracy by gender

5. **error_type_analysis.png**
   - Accuracy by error type
   - Error type distribution (pie chart)
   - Error types × Medical specialty heatmap
   - Confidence score distribution by error type

### Data Files

- `summary_statistics_YYYYMMDD_HHMMSS.json` - Comprehensive summary statistics
- `detailed_results_YYYYMMDD_HHMMSS.csv` - Full dataset with all extracted features

## Analysis Dimensions

### Medical Specialties

The system automatically classifies cases into:
- Infectious Disease
- Cardiovascular
- Respiratory
- Gastrointestinal
- Neurological
- Musculoskeletal
- Dermatological
- Endocrine/Metabolic
- Genitourinary
- Oncology
- Hematological
- General Medicine

### Demographics

- **Age Groups**: Pediatric (0-17), Young Adult (18-35), Middle-aged (36-55), Older Adult (56+)
- **Gender**: Male, Female, Unknown

### Error Types

Extracted from the validation data (e.g., causalOrganism, treatment, diagnosis, etc.)

## Features

- **Automatic Feature Extraction**: Age, gender, and medical specialty from medical notes
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score per group
- **Cross-Tabulation Analysis**: Multi-dimensional performance breakdowns
- **High-Quality Visualizations**: Publication-ready plots at 300 DPI
- **CSV Export**: Full dataset for further analysis in Excel/R/Python

## Troubleshooting

### Common Issues

1. **No result files found**
   ```
   Error: No prediction results found
   ```
   - Check that result files exist in the specified directory
   - Verify file naming pattern: `result_*.json` or `debate_*.json`

2. **Missing dependencies**
   ```
   ModuleNotFoundError: No module named 'matplotlib'
   ```
   - Install required packages: `pip install -r requirements.txt`

3. **Memory issues on large datasets**
   - Increase SLURM memory allocation: `#SBATCH --mem=32G`
   - Process results in batches if needed

4. **Permission denied**
   ```bash
   chmod +x evaluation/run_error_analysis.sh
   ```

## Example Workflow

```bash
# 1. Run predictions (if not already done)
python evaluation/batch_predict.py

# 2. Run error analysis
python evaluation/error_analysis_with_plots.py

# 3. View results
open evaluation/plots/overall_performance.png
open evaluation/plots/summary_statistics_*.json

# 4. For cluster (SLURM)
sbatch evaluation/run_error_analysis.slurm
```

## Customization

### Adding New Specialties

Edit the `SPECIALTY_KEYWORDS` dictionary in `error_analysis_with_plots.py`:

```python
SPECIALTY_KEYWORDS = {
    'Your Specialty': [
        'keyword1', 'keyword2', 'keyword3'
    ],
    # ... existing specialties
}
```

### Modifying Age Groups

Edit the `AGE_GROUPS` dictionary:

```python
AGE_GROUPS = {
    'Infant (0-2)': (0, 2),
    'Child (3-12)': (3, 12),
    # ... custom age groups
}
```

### Changing Plot Styles

Modify the style settings at the top of the script:

```python
sns.set_style("whitegrid")  # Options: whitegrid, darkgrid, white, dark, ticks
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
```

## Performance

Typical execution times:
- 100 cases: ~10 seconds
- 1,000 cases: ~30 seconds
- 10,000 cases: ~3 minutes

Memory usage:
- ~500 MB for 1,000 cases
- ~2 GB for 10,000 cases

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{bluemed_error_analysis,
  title={Comprehensive Error Analysis for Medical AI Systems},
  author={BLUEmed Team},
  year={2024}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `evaluation/logs/`
3. Open an issue on the project repository
