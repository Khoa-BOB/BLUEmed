# Fairness Evaluation Using Equality Difference (ED) Metrics

## Overview

This document explains the demographic fairness evaluation conducted for the BLUEmed medical error detection system using **Equality Difference (ED)** metrics, as described in Dixon et al. (2018), Park et al. (2018), and Garg et al. (2019).

---

## 1. What is Equality Difference (ED)?

**Equality Difference (ED)** is a fairness metric that measures disparities in classification performance across different demographic groups. It quantifies how much performance rates vary between subgroups and the overall population.

### Formula

For a given performance metric (e.g., False Positive Rate):

```
ED = Σ_{d∈D} |Rate_d - Rate_overall|
```

Where:
- `D` is a demographic factor (e.g., gender, age group)
- `d` is a specific demographic group (e.g., Male, Female)
- `Rate_d` is the performance rate for group d
- `Rate_overall` is the overall performance rate across all groups

### Interpretation

- **Lower ED scores** = Better fairness (less disparity between groups)
- **Higher ED scores** = Worse fairness (more disparity between groups)
- **ED = 0** = Perfect fairness (all groups have identical rates)

---

## 2. Metrics Evaluated

We evaluated four key performance rates that are critical for understanding fairness in imbalanced datasets:

### 2.1 False Positive Rate (FPR)
```
FPR = FP / (FP + TN)
```
- Measures how often the model incorrectly predicts INCORRECT when the note is actually CORRECT
- Important for patient safety: False alarms about errors when none exist
- **Lower is better**

### 2.2 False Negative Rate (FNR)
```
FNR = FN / (FN + TP)
```
- Measures how often the model misses actual errors (predicts CORRECT when it's INCORRECT)
- **Critical for patient safety**: Missing real medical errors
- **Lower is better**

### 2.3 True Positive Rate (TPR) / Recall / Sensitivity
```
TPR = TP / (TP + FN)
```
- Measures how well the model detects actual errors
- Complement of FNR: `TPR = 1 - FNR`
- **Higher is better**

### 2.4 True Negative Rate (TNR) / Specificity
```
TNR = TN / (TN + FP)
```
- Measures how well the model correctly identifies error-free notes
- Complement of FPR: `TNR = 1 - FPR`
- **Higher is better**

---

## 3. Why ED Instead of Just F1 Score?

### Problem with F1 Score Alone
- **F1 score** is a global metric that averages precision and recall
- It doesn't reveal **disparities between demographic groups**
- A model can have good overall F1 but still be unfair to specific groups

### Advantages of ED Metrics
1. **Group-level fairness**: Detects if certain demographic groups are treated unfairly
2. **Multiple perspectives**: Evaluates FPR, FNR, TPR, TNR separately
3. **Imbalanced data handling**: ED works well with class imbalance by examining rates (proportions) rather than raw counts
4. **Safety-critical**: In medical AI, fairness in FNR (missing errors) is crucial for all patient groups

---

## 4. Methodology

### Step 1: Data Preparation
1. Loaded 597 prediction results from `logs/debates/result_*.json`
2. Extracted demographics (age, gender) from medical notes
3. Matched predictions with ground truth labels

### Step 2: Demographic Grouping
**Age Groups:**
- Pediatric (0-17): 78 cases
- Young Adult (18-35): 201 cases
- Middle-aged (36-55): 129 cases
- Older Adult (56+): 155 cases
- Unknown: 34 cases

**Gender Groups:**
- Male: 331 cases
- Female: 260 cases
- Unknown: 6 cases

### Step 3: Confusion Matrix Calculation
For each demographic group, we calculated:
- **TP (True Positives)**: Correctly identified INCORRECT notes
- **TN (True Negatives)**: Correctly identified CORRECT notes
- **FP (False Positives)**: Incorrectly flagged CORRECT notes as INCORRECT
- **FN (False Negatives)**: Missed actual INCORRECT notes

### Step 4: Rate Calculation
For each group, calculated TPR, TNR, FPR, FNR using confusion matrix values.

### Step 5: ED Calculation
For each metric and demographic dimension, calculated:
```
ED = |Rate_group1 - Rate_overall| + |Rate_group2 - Rate_overall| + ...
```

---

## 5. Results Summary

### Overall Statistics
- **Total Cases Evaluated**: 597
- **Mean ED across all metrics**: 0.3023
- **ED Range**: 0.0501 to 0.4826

### 5.1 Fairness by Age Group

| Metric | ED Score | Interpretation |
|--------|----------|----------------|
| **FPR** | **0.2133** | Moderate fairness - False positive rates fairly consistent across age groups |
| **FNR** | **0.4826** | ⚠️ Poor fairness - Large disparity in missing errors across age groups |
| **TPR** | **0.4826** | ⚠️ Poor fairness - Detection rates vary significantly by age |
| **TNR** | **0.2133** | Moderate fairness - Correctly identifying error-free notes fairly equal |

**Key Finding for Age:**
- **Pediatric patients (0-17)** have highest TPR (82.93%) but also highest FPR (54.05%)
  - Model is more "aggressive" in detecting errors in pediatric cases
  - More false alarms but also catches more real errors
- **Unknown age group** has very high TPR (88.24%) but small sample size (n=34)
- **Middle-aged (36-55)** patients have lowest TPR (60.29%)
  - Model misses more errors in this age group

### 5.2 Fairness by Gender

| Metric | ED Score | Interpretation |
|--------|----------|----------------|
| **FPR** | **0.4631** | ⚠️ Poor fairness - High disparity in false positive rates |
| **FNR** | **0.0501** | ✅ Excellent fairness - Very consistent in missing error rates |
| **TPR** | **0.0501** | ✅ Excellent fairness - Very consistent in detecting errors |
| **TNR** | **0.4631** | ⚠️ Poor fairness - Large variation in correctly identifying error-free notes |

**Key Finding for Gender:**
- **Female patients**: Slightly higher TPR (69.85% vs 65.70% for males)
  - Model slightly better at detecting errors in female patient notes
- **Male patients**: Similar FPR (45.91% vs 44.35% for females)
- **Unknown gender** (n=6): TNR of 100% skews the ED score (small sample size issue)
- Overall, gender disparity is **much lower than age disparity**

---

## 6. Clinical Implications

### Age-Related Concerns
1. **Pediatric Over-Detection**: Higher FPR in pediatric cases means more false alarms
   - Could lead to unnecessary investigations
   - But also catches more real errors (safety trade-off)

2. **Middle-Aged Under-Detection**: Lower TPR means missing more errors
   - This age group may have more complex medical presentations
   - Need to investigate why model performs worse

### Gender-Related Concerns
1. **Relatively Fair**: Small differences in TPR/FNR between genders is positive
2. **FPR Disparity**: Different false positive rates could affect clinical workflow
   - More investigations triggered for one gender vs another

---

## 7. Visualizations Generated

### 7.1 ED Heatmap (`fairness_ed_heatmap.png`)
- Shows ED scores for all metrics × dimensions in a color-coded matrix
- Red = higher disparity, Yellow = lower disparity
- Quick visual comparison of fairness across all factors

### 7.2 Rate Comparison Plots (`fairness_rates_age_group.png`, `fairness_rates_gender.png`)
- Bar charts showing TPR, TNR, FPR, FNR for each demographic group
- Red dashed line = overall rate
- Bars above/below line show disparity
- Sample sizes (n) shown on each bar

### 7.3 Confusion Matrix Breakdown (`confusion_breakdown_age_group.png`, `confusion_breakdown_gender.png`)
- Stacked bar charts showing TP, TN, FP, FN counts
- Visual comparison of error types across groups
- Absolute counts rather than rates

### 7.4 ED Bar Charts (`ed_barchart_age_group.png`, `ed_barchart_gender.png`)
- Direct comparison of ED scores for all four metrics
- Lower bars = more fair
- Color-coded by metric type

---

## 8. Comparison to Standard Metrics

### What Traditional Metrics Show
Running `demographic_analysis.py` gives:
- Overall Accuracy: High
- Overall F1 Score: Good
- **But**: Doesn't reveal age/gender disparities

### What ED Metrics Reveal
- **Age disparity in error detection** (ED = 0.48 for TPR/FNR)
- **Gender disparity in false positives** (ED = 0.46 for FPR/TNR)
- Specific groups that are under-served or over-flagged

---

## 9. Recommendations

### Immediate Actions
1. **Investigate middle-aged group**: Why is TPR lowest for ages 36-55?
   - Review misclassified cases
   - Check if medical notes differ in complexity or terminology

2. **Audit pediatric predictions**: High FPR may need adjustment
   - Review false positives to identify patterns
   - Consider age-specific thresholds

3. **Monitor unknown groups**: Small sample sizes (n=34, n=6) cause instability
   - Improve demographic extraction from notes
   - Ensure consistent demographic labeling

### Long-term Improvements
1. **Age-stratified training**: Consider balancing training data by age
2. **Calibration by demographic**: Adjust confidence thresholds per group
3. **Continuous monitoring**: Track ED metrics in production
4. **Expand evaluation**: Add race/ethnicity when data available

---

## 10. Technical Details

### Implementation
- **Script**: `evaluation/fairness_evaluation.py`
- **Dependencies**: numpy, matplotlib, sklearn (optional)
- **Input**: Prediction results from `logs/debates/result_*.json`
- **Output**:
  - JSON: Full results with all calculations
  - CSV: Summary table of ED scores
  - PNG: 7 visualization plots at 300 DPI

### Running the Evaluation
```bash
python3 evaluation/fairness_evaluation.py \
  --results-dir logs/debates \
  --output-dir evaluation/fairness
```

### Output Files
- `fairness_evaluation_YYYYMMDD_HHMMSS.json` - Complete results
- `fairness_summary_YYYYMMDD_HHMMSS.csv` - ED scores table
- `fairness_ed_heatmap.png` - Heatmap of all ED scores
- `fairness_rates_*.png` - Rate comparison plots
- `confusion_breakdown_*.png` - Confusion matrix breakdowns
- `ed_barchart_*.png` - ED score bar charts

---

## 11. Limitations

1. **Small sample sizes**: Unknown gender (n=6) and Unknown age (n=34) groups
   - Results less reliable for these groups
   - ED scores inflated by small denominators

2. **Demographic extraction**: Based on text parsing
   - May miss or misclassify demographics
   - "Unknown" category captures extraction failures

3. **Binary classification**: Only evaluates CORRECT vs INCORRECT
   - Doesn't capture error type fairness
   - Could extend to multi-class ED

4. **Intersectionality**: Doesn't evaluate age × gender interactions
   - Future work: ED for age-gender subgroups
   - Would require larger dataset

---

## 12. References

- **Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018)**. Measuring and mitigating unintended bias in text classification. *Proceedings of AAAI/ACM Conference on AI, Ethics, and Society*.

- **Park, J. H., Shin, J., & Fung, P. (2018)**. Reducing gender bias in abusive language detection. *arXiv preprint arXiv:1808.07231*.

- **Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2019)**. Counterfactual fairness in text classification through robustness. *Proceedings of AAAI/ACM Conference on AI, Ethics, and Society*.

---

## Summary

We successfully implemented and evaluated demographic fairness using Equality Difference (ED) metrics. The evaluation revealed:

✅ **Gender fairness is good**: ED = 0.05 for TPR/FNR (error detection)
⚠️ **Age fairness needs improvement**: ED = 0.48 for TPR/FNR
⚠️ **False positive disparity**: Gender ED = 0.46 for FPR/TNR

**Next steps**: Investigate age-related disparities and consider age-stratified model improvements.
