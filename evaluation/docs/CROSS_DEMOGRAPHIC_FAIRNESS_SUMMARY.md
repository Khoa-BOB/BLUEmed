# Cross-Demographic Fairness Analysis: Gender × Age

## Overview

This document presents the **cross-demographic (intersectional) fairness evaluation** for the BLUEmed system, examining how performance varies across **Gender × Age** combinations.

---

## Key Findings

### Cross-Demographic ED Scores (Gender × Age)

| Metric | Gender ED | Age ED | **Cross ED** | Interpretation |
|--------|-----------|---------|--------------|----------------|
| **TPR** | 0.050 | 0.483 | **1.288** | Cross-demographic disparity is **2.7× worse** than age alone |
| **TNR** | 0.463 | 0.213 | **1.018** | Cross-demographic disparity is **2.2× worse** than gender alone |
| **FPR** | 0.463 | 0.213 | **1.018** | Similar pattern to TNR |
| **FNR** | 0.050 | 0.483 | **1.288** | Cross-demographic disparity matches TPR pattern |

**Critical Finding**: When we combine gender and age, we see **much larger disparities** than when examining either dimension alone. This reveals **intersectional bias** that was hidden in single-dimension analysis.

---

## Detailed Cross-Demographic Results

### 1. True Positive Rate (TPR) - Error Detection Rate

**Overall TPR: 67.52%**

**Cross-Demographic ED: 1.288 (vs 0.483 for age, 0.050 for gender)**

#### Top Performers (Highest Error Detection):
1. **Male × Unknown**: 100.00% TPR (n=20)
   - Perfect error detection (but small sample size)
   - Diff from overall: +32.48%

2. **Female × Pediatric (0-17)**: 86.67% TPR (n=29)
   - Excellent detection in female children
   - Diff from overall: +19.14%

3. **Female × Older Adult (56+)**: 83.33% TPR (n=56)
   - Strong performance for older women
   - Diff from overall: +15.81%

4. **Male × Pediatric (0-17)**: 80.00% TPR (n=47)
   - Good detection for male children
   - Diff from overall: +12.48%

#### Bottom Performers (Lowest Error Detection):
1. **Male × Middle-aged (36-55)**: 56.94% TPR (n=71)
   - **Worst performance**: Misses 43% of errors
   - Diff from overall: -10.58%
   - **28.73% gap** vs Female × Pediatric

2. **Male × Older Adult (56+)**: 61.62% TPR (n=98)
   - Below average for older men
   - Diff from overall: -5.90%

3. **Female × Young Adult (18-35)**: 60.71% TPR (n=106)
   - Lower detection for young adult women
   - Diff from overall: -6.81%

---

### 2. False Negative Rate (FNR) - Miss Rate

**Overall FNR: 32.48%**

**Cross-Demographic ED: 1.288**

#### Best (Lowest Miss Rate):
1. **Male × Unknown**: 0.00% FNR (n=20)
   - Never misses errors (small sample)

2. **Female × Pediatric (0-17)**: 13.33% FNR (n=29)
   - Rarely misses errors in female children

3. **Female × Older Adult (56+)**: 16.67% FNR (n=56)

#### Worst (Highest Miss Rate):
1. **Male × Middle-aged (36-55)**: 43.06% FNR (n=71)
   - **Misses nearly half of all errors**
   - This is a **critical patient safety issue**

2. **Male × Older Adult (56+)**: 38.38% FNR (n=98)
   - Also high miss rate

---

### 3. False Positive Rate (FPR) - False Alarm Rate

**Overall FPR: 44.76%**

**Cross-Demographic ED: 1.018**

#### Lowest False Alarms:
1. **Male × Unknown**: 18.18% FPR (n=20)
   - Fewest false alarms

2. **Female × Young Adult (18-35)**: 35.56% FPR (n=106)

#### Highest False Alarms:
1. **Female × Pediatric (0-17)**: 60.00% FPR (n=29)
   - High false alarm rate
   - Trade-off: High detection but also high false alarms

2. **Male × Pediatric (0-17)**: 52.38% FPR (n=47)
   - Similar pattern

---

## Intersectional Insights

### 1. **Pediatric Female Patients: Over-Detection Pattern**
- **Highest TPR** (86.67%) but also **highest FPR** (60%)
- Model is "aggressive" - catches most errors but triggers many false alarms
- **Clinical Impact**: More investigations but safer (doesn't miss errors)

### 2. **Middle-Aged Male Patients: Under-Detection Crisis** ⚠️
- **Lowest TPR** (56.94%) and **highest FNR** (43.06%)
- Model consistently misses errors in this demographic
- **Clinical Impact**: Serious patient safety concern
- **28.73% performance gap** vs best group

### 3. **Gender × Age Interaction Effects**
Comparing male vs female within same age groups:

| Age Group | Male TPR | Female TPR | Difference |
|-----------|----------|------------|------------|
| Pediatric (0-17) | 80.00% | 86.67% | +6.67% (Female better) |
| Young Adult (18-35) | 65.96% | 60.71% | -5.25% (Male better) |
| Middle-aged (36-55) | 56.94% | 62.07% | +5.13% (Female better) |
| Older Adult (56+) | 61.62% | 83.33% | **+21.71%** (Female much better) |

**Key Pattern**:
- Female patients have **much better** error detection in pediatric and older adult groups
- Male patients have slightly better detection in young adults
- The gender gap **widens with age** (especially 56+)

---

## Why Cross-Demographic ED is Higher

### Mathematical Explanation

**Single-dimension ED** averages out subgroup differences:
- When we only look at "Male" overall, we combine pediatric males (80% TPR) with middle-aged males (57% TPR)
- The average hides the large variation within males

**Cross-demographic ED** exposes hidden disparities:
- Now we see that "Male × Middle-aged" performs **24% worse** than "Male × Pediatric"
- These within-gender differences weren't visible before

### Analogy
Think of grades in a class:
- **Single-dimension**: "Class A average: 75%, Class B average: 78%" (small difference)
- **Cross-demographic**: "Class A freshmen: 90%, Class A seniors: 60%, Class B freshmen: 80%, Class B seniors: 76%"
- The cross-view reveals that Class A seniors are struggling, which was hidden in the overall average

---

## Clinical Implications

### Urgent Actions Required

1. **Investigate Middle-Aged Male Under-Detection**
   - 43% miss rate is unacceptable
   - Review medical note characteristics for this group
   - May need age-gender-specific model calibration

2. **Audit Pediatric Female False Positives**
   - 60% false alarm rate strains clinical resources
   - Review what triggers false alarms
   - Consider adjusting decision threshold for this group

3. **Leverage Older Adult Female Success**
   - 83% TPR with moderate FPR (44%)
   - Understand what works well for this group
   - Apply lessons to other demographics

### Fairness Considerations

**Equity Question**: Should all groups have equal error detection rates?

**Arguments for equal rates:**
- Ethical fairness: Every patient deserves equal protection
- Legal compliance: Avoid discriminatory healthcare AI

**Arguments for different rates:**
- Medical complexity varies by demographics
- Some age groups may have more ambiguous presentations
- Trade-offs between TPR and FPR differ by clinical context

**Recommendation**: Aim for **bounded disparity**
- No group should have <60% TPR (currently Male × Middle-aged = 57%)
- No group should have >50% FPR (currently Female × Pediatric = 60%)
- Target: Keep cross-demographic ED < 0.5 for TPR/FNR

---

## Comparison: Simple vs Cross-Demographic Analysis

### What We Missed with Simple Analysis

**Before (Gender only):**
- "Gender is fair: TPR ED = 0.050"
- Concluded: No gender bias in error detection

**After (Gender × Age):**
- "Older female patients: 83% TPR"
- "Older male patients: 62% TPR"
- **21% gap within the same age group!**
- The gender effect depends on age (interaction effect)

### Statistical Significance

The cross-demographic ED is **2.5-3× larger** than single-dimension ED:
- This is not just noise - it reveals real patterns
- 10+ groups with n ≥ 10 provide reliable estimates
- Consistent patterns across TPR/TNR/FPR/FNR

---

## Visualizations Generated

### 1. Cross-Demographic Heatmaps (4 files)
- `cross_demographic_TPR_heatmap.png`
- `cross_demographic_TNR_heatmap.png`
- `cross_demographic_FPR_heatmap.png`
- `cross_demographic_FNR_heatmap.png`

**Shows**: Gender (rows) × Age (columns) with color-coded rates
- Red = Lower rates
- Green = Higher rates
- Darker cells = Larger sample sizes

### 2. ED Comparison Chart
- `ed_comparison_cross_demographic.png`

**Shows**: Side-by-side bars comparing:
- Gender ED (blue)
- Age ED (orange)
- Gender × Age ED (red)

Clearly illustrates that cross-demographic ED is much higher.

---

## Technical Details

### Groups Included (n ≥ 10)
10 groups met the minimum sample size:
- Male × Young Adult (18-35): n=93
- Male × Middle-aged (36-55): n=71
- Male × Older Adult (56+): n=98
- Male × Unknown: n=20
- Male × Pediatric (0-17): n=47
- Female × Young Adult (18-35): n=106
- Female × Middle-aged (36-55): n=57
- Female × Older Adult (56+): n=56
- Female × Pediatric (0-17): n=29
- Female × Unknown: n=12

### Groups Excluded (n < 10)
3 groups excluded due to small sample size:
- Unknown gender × various age groups

### Calculation Example

For TPR cross-demographic ED:
```
Overall TPR = 67.52%

Male × Young Adult: 65.96% → |65.96 - 67.52| = 1.56%
Male × Middle-aged: 56.94% → |56.94 - 67.52| = 10.58%
...
[Sum all 10 groups' absolute differences]
...
Cross-Demographic TPR ED = 1.288 (128.8%)
```

This is **2.67× larger** than Age-only ED (48.3%) and **25.8× larger** than Gender-only ED (5.0%).

---

## Recommendations

### Short-term (1-3 months)
1. **Flag high-risk predictions** for middle-aged male patients
   - Add "confidence boost" or secondary review
   - Investigate why model struggles with this group

2. **Calibrate thresholds by demographic**
   - Lower threshold for male × middle-aged (catch more errors)
   - Raise threshold for female × pediatric (reduce false alarms)

3. **Conduct error analysis**
   - Review misclassified cases by demographic
   - Identify patterns in medical notes

### Medium-term (3-6 months)
1. **Collect more data** for underrepresented groups
   - Especially male × middle-aged cases
   - Balance training data by gender × age

2. **Train demographic-aware model**
   - Include gender and age as input features
   - Use demographic stratified sampling

3. **Implement fairness constraints**
   - Add ED penalty to loss function during training
   - Target: Cross-demographic ED < 0.5

### Long-term (6-12 months)
1. **Continuous monitoring**
   - Track cross-demographic ED in production
   - Alert if any group's TPR drops below threshold

2. **Expand analysis**
   - Add race/ethnicity when data available
   - Consider 3-way intersections (gender × age × race)

3. **Clinical validation study**
   - Prospectively evaluate fairness metrics
   - Measure real-world impact on patient outcomes

---

## Conclusion

Cross-demographic analysis revealed **critical intersectional biases** that were invisible in single-dimension fairness evaluation:

✅ **Gender alone**: Fair (ED = 0.05)
⚠️ **Age alone**: Moderate disparity (ED = 0.48)
🚨 **Gender × Age**: **Large disparity (ED = 1.29)**

**Most critical finding**: **Male middle-aged patients** have a 57% error detection rate, missing 43% of errors - **the worst performance of any demographic group**.

**Action required**: Immediate investigation and model adjustment for this high-risk group.

---

## Files Generated

- **JSON**: `fairness_evaluation_20260119_082205.json` (complete results)
- **CSV**: `fairness_summary_20260119_082205.csv` (ED scores table)
- **Heatmaps**: 4 cross-demographic heatmaps (TPR, TNR, FPR, FNR)
- **Comparison**: ED comparison chart (simple vs cross-demographic)
- **Previous files**: All original single-dimension visualizations

Total: **12 visualization files** covering all fairness dimensions.
