#!/usr/bin/env python3
"""
Comprehensive Error Analysis for Medical Specialty Classification
Analyzes errors by specialty, error type, demographics (age, gender), and cross-tabulations.
Generates comprehensive visualization plots.

Usage:
    python evaluation/error_analysis_with_plots.py
    python evaluation/error_analysis_with_plots.py --results-dir logs/debates --output-dir evaluation/plots
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ComprehensiveErrorAnalyzer:
    """Comprehensive error analysis with demographic and specialty breakdowns."""

    # Age group definitions
    AGE_GROUPS = {
        'Pediatric (0-17)': (0, 17),
        'Young Adult (18-35)': (18, 35),
        'Middle-aged (36-55)': (36, 55),
        'Older Adult (56+)': (56, 200)
    }

    # Medical specialty classification keywords
    SPECIALTY_KEYWORDS = {
        'Infectious Disease': [
            'infection', 'bacterial', 'viral', 'virus', 'bacteria', 'sepsis', 'septic',
            'pneumonia', 'tuberculosis', 'HIV', 'AIDS', 'hepatitis', 'meningitis',
            'gonorrhea', 'chlamydia', 'syphilis', 'herpes', 'influenza', 'flu',
            'streptococcus', 'staphylococcus', 'e. coli', 'salmonella', 'malaria',
            'typhoid', 'cholera', 'UTI', 'urinary tract infection', 'cellulitis',
            'abscess', 'antimicrobial', 'antibiotic', 'pathogen', 'organism'
        ],
        'Cardiovascular': [
            'heart', 'cardiac', 'cardiovascular', 'myocardial', 'infarction',
            'arrhythmia', 'atrial', 'ventricular', 'hypertension', 'hypotension',
            'angina', 'coronary', 'aortic', 'valve', 'murmur', 'palpitation',
            'tachycardia', 'bradycardia', 'chest pain', 'EKG', 'ECG', 'echocardiogram',
            'warfarin', 'anticoagulant'
        ],
        'Respiratory': [
            'lung', 'pulmonary', 'respiratory', 'bronchitis', 'asthma', 'COPD',
            'emphysema', 'cough', 'dyspnea', 'shortness of breath', 'wheeze',
            'stridor', 'pneumothorax', 'pleural', 'bronchial', 'alveolar'
        ],
        'Gastrointestinal': [
            'stomach', 'intestinal', 'bowel', 'colon', 'gastric', 'hepatic', 'liver',
            'pancreatic', 'pancreatitis', 'appendicitis', 'cholecystitis', 'diarrhea',
            'vomiting', 'nausea', 'abdominal pain', 'GI', 'gastrointestinal', 'ulcer'
        ],
        'Neurological': [
            'brain', 'neurological', 'seizure', 'epilepsy', 'stroke', 'TIA',
            'headache', 'migraine', 'meningitis', 'encephalitis', 'neuropathy',
            'paralysis', 'weakness', 'numbness', 'tingling', 'consciousness'
        ],
        'Musculoskeletal': [
            'bone', 'joint', 'muscle', 'arthritis', 'fracture', 'osteoporosis',
            'back pain', 'knee', 'hip', 'shoulder', 'spine', 'orthopedic',
            'arthroplasty', 'prosthesis'
        ],
        'Dermatological': [
            'skin', 'rash', 'lesion', 'dermatitis', 'eczema', 'psoriasis',
            'wound', 'ulcer', 'blister', 'erythema', 'pruritus', 'itching',
            'bruise', 'necrosis'
        ],
        'Endocrine/Metabolic': [
            'diabetes', 'diabetic', 'thyroid', 'insulin', 'glucose', 'metabolic',
            'hormone', 'endocrine', 'obesity', 'hypoglycemia', 'hyperglycemia'
        ],
        'Genitourinary': [
            'kidney', 'renal', 'bladder', 'urinary', 'prostate', 'testicular',
            'ovarian', 'uterine', 'vaginal', 'genital', 'genitourinary', 'STI', 'STD'
        ],
        'Oncology': [
            'cancer', 'tumor', 'malignant', 'benign', 'carcinoma', 'sarcoma',
            'lymphoma', 'leukemia', 'metastasis', 'oncology', 'chemotherapy'
        ],
        'Hematological': [
            'blood', 'anemia', 'leukocyte', 'hemoglobin', 'platelet', 'coagulation',
            'bleeding', 'clot', 'thrombosis', 'hematology'
        ]
    }

    def __init__(self, validation_file: str = "test_data/validation.json",
                 results_dir: str = "logs/debates"):
        """Initialize analyzer."""
        self.validation_file = Path(validation_file)
        self.results_dir = Path(results_dir)
        self.validation_data = {}
        self.results_data = {}
        self.merged_data = []

    def load_data(self) -> int:
        """Load validation data and prediction results."""
        print(f"Loading validation data from {self.validation_file}...")
        with open(self.validation_file, 'r', encoding='utf-8') as f:
            validation_list = json.load(f)

        for case in validation_list:
            case_id = case.get('id')
            if case_id:
                self.validation_data[case_id] = case
        print(f"  Loaded {len(self.validation_data)} validation cases")

        print(f"Loading prediction results from {self.results_dir}...")
        result_files = list(self.results_dir.glob("result_*.json"))

        # Also check for debate_*.json files
        if not result_files:
            result_files = list(self.results_dir.glob("debate_*.json"))

        for filepath in result_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                case_id = result.get('case_id')
                if case_id:
                    self.results_data[case_id] = result
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

        print(f"  Loaded {len(self.results_data)} prediction results")
        return len(self.results_data)

    def extract_age(self, text: str) -> Optional[int]:
        """Extract age from medical note text."""
        patterns = [
            r'(\d{1,3})[-\s]?year[-\s]?old',
            r'aged?\s*(\d{1,3})',
            r'(\d{1,3})\s*years?\s*old',
            r'(\d{1,3})\s*yo\b',
            r'(\d{1,3})\s*y/?o\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 120:
                    return age
        return None

    def extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from medical note text."""
        text_lower = text.lower()

        male_patterns = [
            r'\b(man|male|boy|gentleman|he|his|him)\b',
            r'\b(\d+[-\s]?year[-\s]?old)\s+(man|male|boy)\b',
        ]
        female_patterns = [
            r'\b(woman|female|girl|lady|she|her)\b',
            r'\b(\d+[-\s]?year[-\s]?old)\s+(woman|female|girl)\b',
        ]

        male_count = sum(len(re.findall(p, text_lower)) for p in male_patterns)
        female_count = sum(len(re.findall(p, text_lower)) for p in female_patterns)

        if re.search(r'\d+[-\s]?year[-\s]?old\s+(woman|female|girl)', text_lower):
            return 'Female'
        if re.search(r'\d+[-\s]?year[-\s]?old\s+(man|male|boy)', text_lower):
            return 'Male'

        if female_count > male_count:
            return 'Female'
        elif male_count > female_count:
            return 'Male'

        return None

    def get_age_group(self, age: Optional[int]) -> str:
        """Map age to age group."""
        if age is None:
            return 'Unknown'
        for group_name, (min_age, max_age) in self.AGE_GROUPS.items():
            if min_age <= age <= max_age:
                return group_name
        return 'Unknown'

    def classify_specialty(self, text: str) -> str:
        """Classify medical specialty based on keywords."""
        text_lower = text.lower()
        scores = {}

        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[specialty] = score

        if scores:
            return max(scores, key=scores.get)
        return 'General Medicine'

    def merge_data(self) -> List[Dict]:
        """Merge validation data with prediction results and extract features."""
        self.merged_data = []

        for case_id, result in self.results_data.items():
            val_case = self.validation_data.get(case_id, {})
            text = val_case.get('text', result.get('medical_note', ''))

            # Clean text if it has quotes
            if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
                text = text[1:-1]

            age = self.extract_age(text)
            gender = self.extract_gender(text)
            age_group = self.get_age_group(age)
            specialty = self.classify_specialty(text)

            ground_truth = result.get('ground_truth', val_case.get('label'))
            predicted_label = result.get('predicted_label', result.get('final_answer'))

            # Convert string predictions to int if needed
            if isinstance(predicted_label, str):
                predicted_label = 1 if predicted_label.upper() == 'INCORRECT' else 0
            if isinstance(ground_truth, str):
                ground_truth = 1 if ground_truth.upper() == 'INCORRECT' else 0

            record = {
                'case_id': case_id,
                'age': age,
                'age_group': age_group,
                'gender': gender if gender else 'Unknown',
                'specialty': specialty,
                'ground_truth': ground_truth,
                'error_type': result.get('error_type', val_case.get('error_type', 'NA')),
                'predicted_label': predicted_label,
                'confidence_score': result.get('confidence_score'),
                'confidence_normalized': result.get('confidence_normalized'),
                'is_correct': ground_truth == predicted_label if ground_truth is not None and predicted_label is not None else None,
                'execution_time': result.get('execution_time'),
                'winner': result.get('winner', 'Unknown'),
                'text_length': len(text)
            }

            self.merged_data.append(record)

        print(f"Merged {len(self.merged_data)} records")
        return self.merged_data

    def create_dataframe(self) -> pd.DataFrame:
        """Convert merged data to pandas DataFrame."""
        df = pd.DataFrame(self.merged_data)
        return df

    def filter_valid_data(self, df: pd.DataFrame, exclude_na_errors: bool = False) -> pd.DataFrame:
        """
        Filter dataframe for valid predictions.

        Args:
            df: Input dataframe
            exclude_na_errors: If True, exclude cases with error_type == 'NA' or 'N/A'

        Returns:
            Filtered dataframe
        """
        # Filter for valid predictions (non-null is_correct)
        valid_df = df[df['is_correct'].notna()].copy()

        # Optionally exclude NA error types (these are correct cases without errors)
        if exclude_na_errors:
            valid_df = valid_df[~valid_df['error_type'].isin(['NA', 'N/A', 'na', 'n/a'])].copy()

        return valid_df

    def consolidate_others(self, df: pd.DataFrame, column: str, threshold: float = 0.05) -> pd.DataFrame:
        """
        Consolidate categories with less than threshold% into 'Others'.

        Args:
            df: Input dataframe
            column: Column name to consolidate
            threshold: Percentage threshold (0.05 = 5%)

        Returns:
            DataFrame with consolidated categories
        """
        df_copy = df.copy()
        total_count = len(df_copy)
        value_counts = df_copy[column].value_counts()

        # Find categories below threshold
        small_categories = value_counts[value_counts < (total_count * threshold)].index

        # Replace with 'Others'
        df_copy[column] = df_copy[column].apply(
            lambda x: 'Others' if x in small_categories else x
        )

        return df_copy

    def plot_overall_performance(self, df: pd.DataFrame, output_dir: Path):
        """Plot overall model performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Filter valid predictions (excluding NA error types)
        valid_df = self.filter_valid_data(df, exclude_na_errors=True)

        # 1. Confusion Matrix
        if len(valid_df) > 0:
            cm = confusion_matrix(valid_df['ground_truth'], valid_df['predicted_label'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('True Label')
            axes[0, 0].set_xlabel('Predicted Label')
            axes[0, 0].set_xticklabels(['Correct', 'Incorrect'])
            axes[0, 0].set_yticklabels(['Correct', 'Incorrect'])

        # 2. Accuracy by Ground Truth
        acc_by_label = valid_df.groupby('ground_truth')['is_correct'].agg(['sum', 'count'])
        acc_by_label['accuracy'] = acc_by_label['sum'] / acc_by_label['count']
        axes[0, 1].bar(['Correct Cases', 'Incorrect Cases'], acc_by_label['accuracy'].values,
                       color=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Accuracy by Ground Truth Label', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(acc_by_label['accuracy'].values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        # 3. Performance Metrics
        metrics = {
            'Accuracy': accuracy_score(valid_df['ground_truth'], valid_df['predicted_label']),
            'Precision': precision_score(valid_df['ground_truth'], valid_df['predicted_label'], zero_division=0),
            'Recall': recall_score(valid_df['ground_truth'], valid_df['predicted_label'], zero_division=0),
            'F1-Score': f1_score(valid_df['ground_truth'], valid_df['predicted_label'], zero_division=0)
        }
        axes[1, 0].barh(list(metrics.keys()), list(metrics.values()), color='skyblue')
        axes[1, 0].set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_xlim([0, 1])
        for i, (k, v) in enumerate(metrics.items()):
            axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

        # 4. Prediction Distribution
        pred_dist = valid_df['predicted_label'].value_counts()
        axes[1, 1].pie(pred_dist.values, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: overall_performance.png")

    def plot_specialty_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot accuracy of error detection across medical specialties with bar+line format."""
        # Filter valid predictions (keep NA)
        valid_df = self.filter_valid_data(df, exclude_na_errors=False)

        # Consolidate specialties <5% into Others
        valid_df_consolidated = self.consolidate_others(valid_df, 'specialty', threshold=0.05)

        # Calculate metrics by specialty
        specialty_stats = valid_df_consolidated.groupby('specialty').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        specialty_stats.columns = ['specialty', 'accuracy', 'count']

        # Sort by accuracy descending
        specialty_stats = specialty_stats.sort_values('accuracy', ascending=False)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Bar chart for sample size (muted sky blue for regular, grey for Others)
        x_pos = range(len(specialty_stats))
        colors = ['#87CEEB' if spec != 'Others' else '#808080' for spec in specialty_stats['specialty']]
        bars = ax1.bar(x_pos, specialty_stats['count'], color=colors, alpha=0.7, label='Sample Size')
        ax1.set_xlabel('Medical Specialty', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(specialty_stats['specialty'], rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='black')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, specialty_stats['count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

        # Line plot for accuracy (red)
        ax2 = ax1.twinx()
        line = ax2.plot(x_pos, specialty_stats['accuracy'] * 100,
                       color='red', marker='o', linewidth=2.5, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 100])

        # Add accuracy labels on line
        for i, (x, acc) in enumerate(zip(x_pos, specialty_stats['accuracy'])):
            ax2.text(x, acc * 100 + 2, f'{acc*100:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

        # Title and legend
        plt.title('Accuracy of Error Detection Across Medical Specialties',
                 fontsize=14, fontweight='bold', pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'specialty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: specialty_analysis.png")

    def plot_patient_population_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot accuracy of error detection across patient populations with bar+line format."""
        # Filter valid predictions (keep NA)
        valid_df = self.filter_valid_data(df, exclude_na_errors=False)

        # Consolidate age groups <5% into Others
        valid_df_consolidated = self.consolidate_others(valid_df, 'age_group', threshold=0.05)

        # Calculate metrics by age group
        age_stats = valid_df_consolidated.groupby('age_group').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        age_stats.columns = ['age_group', 'accuracy', 'count']

        # Sort by accuracy descending
        age_stats = age_stats.sort_values('accuracy', ascending=False)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Bar chart for sample size (muted sky blue for regular, grey for Others)
        x_pos = range(len(age_stats))
        colors = ['#87CEEB' if ag != 'Others' else '#808080' for ag in age_stats['age_group']]
        bars = ax1.bar(x_pos, age_stats['count'], color=colors, alpha=0.7, label='Sample Size')
        ax1.set_xlabel('Patient Population', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(age_stats['age_group'], rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='black')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, age_stats['count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

        # Line plot for accuracy (red)
        ax2 = ax1.twinx()
        line = ax2.plot(x_pos, age_stats['accuracy'] * 100,
                       color='red', marker='o', linewidth=2.5, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 100])

        # Add accuracy labels on line
        for i, (x, acc) in enumerate(zip(x_pos, age_stats['accuracy'])):
            ax2.text(x, acc * 100 + 2, f'{acc*100:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

        # Title and legend
        plt.title('Accuracy of Error Detection Across Patient Populations',
                 fontsize=14, fontweight='bold', pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'patient_population_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: patient_population_analysis.png")

    def plot_demographic_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot error analysis by demographics (age, gender)."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Filter valid predictions (excluding NA error types)
        valid_df = self.filter_valid_data(df, exclude_na_errors=True)

        # 1. Accuracy by Age Group
        age_acc = valid_df.groupby('age_group')['is_correct'].agg(['mean', 'count'])
        age_order = ['Pediatric (0-17)', 'Young Adult (18-35)', 'Middle-aged (36-55)', 'Older Adult (56+)', 'Unknown']
        age_acc = age_acc.reindex([ag for ag in age_order if ag in age_acc.index])

        axes[0, 0].bar(range(len(age_acc)), age_acc['mean'].values, color='mediumseagreen')
        axes[0, 0].set_title('Accuracy by Age Group', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(age_acc)))
        axes[0, 0].set_xticklabels(age_acc.index, rotation=45, ha='right')
        axes[0, 0].set_ylim([0, 1])
        for i, (idx, row) in enumerate(age_acc.iterrows()):
            axes[0, 0].text(i, row['mean'] + 0.02, f"{row['mean']:.3f}\n(n={int(row['count'])})",
                           ha='center', fontsize=9)

        # 2. Accuracy by Gender
        gender_acc = valid_df.groupby('gender')['is_correct'].agg(['mean', 'count'])
        colors = {'Male': 'steelblue', 'Female': 'hotpink', 'Unknown': 'gray'}
        bar_colors = [colors.get(g, 'gray') for g in gender_acc.index]

        axes[0, 1].bar(range(len(gender_acc)), gender_acc['mean'].values, color=bar_colors)
        axes[0, 1].set_title('Accuracy by Gender', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(range(len(gender_acc)))
        axes[0, 1].set_xticklabels(gender_acc.index, rotation=0)
        axes[0, 1].set_ylim([0, 1])
        for i, (idx, row) in enumerate(gender_acc.iterrows()):
            axes[0, 1].text(i, row['mean'] + 0.02, f"{row['mean']:.3f}\n(n={int(row['count'])})",
                           ha='center', fontsize=9)

        # 3. Age Distribution
        age_dist = df['age'].dropna()
        axes[0, 2].hist(age_dist, bins=20, color='mediumpurple', edgecolor='black')
        axes[0, 2].set_title('Age Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(age_dist.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age_dist.mean():.1f}')
        axes[0, 2].legend()

        # 4. Cross-tab: Age Group vs Error Type
        age_error = pd.crosstab(valid_df['age_group'], valid_df['error_type'])
        age_error = age_error.reindex([ag for ag in age_order if ag in age_error.index])
        age_error.plot(kind='bar', stacked=False, ax=axes[1, 0], colormap='Set2')
        axes[1, 0].set_title('Error Types by Age Group', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Age Group')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. Cross-tab: Gender vs Error Type
        gender_error = pd.crosstab(valid_df['gender'], valid_df['error_type'])
        gender_error.plot(kind='bar', stacked=False, ax=axes[1, 1], colormap='Set3')
        axes[1, 1].set_title('Error Types by Gender', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Gender')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 1].tick_params(axis='x', rotation=0)

        # 6. Gender Distribution by Age Group
        gender_age = pd.crosstab(df['age_group'], df['gender'])
        gender_age = gender_age.reindex([ag for ag in age_order if ag in gender_age.index])
        gender_age.plot(kind='bar', stacked=True, ax=axes[1, 2], color=['steelblue', 'hotpink', 'gray'])
        axes[1, 2].set_title('Gender Distribution by Age Group', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Age Group')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend(title='Gender', loc='upper right')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: demographic_analysis.png")

    def plot_age_gender_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot accuracy across age×gender combinations with bar+line format."""
        # Filter valid predictions (keep NA)
        valid_df = self.filter_valid_data(df, exclude_na_errors=False)

        # Create age×gender combination
        valid_df['age_gender'] = valid_df['age_group'] + ' - ' + valid_df['gender']

        # Consolidate combinations <5% into Others
        valid_df_consolidated = self.consolidate_others(valid_df, 'age_gender', threshold=0.05)

        # Calculate metrics by age×gender
        age_gender_stats = valid_df_consolidated.groupby('age_gender').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        age_gender_stats.columns = ['age_gender', 'accuracy', 'count']

        # Sort by accuracy descending
        age_gender_stats = age_gender_stats.sort_values('accuracy', ascending=False)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Bar chart for sample size (muted sky blue for regular, grey for Others)
        x_pos = range(len(age_gender_stats))
        colors = ['#87CEEB' if ag != 'Others' else '#808080' for ag in age_gender_stats['age_gender']]
        bars = ax1.bar(x_pos, age_gender_stats['count'], color=colors, alpha=0.7, label='Sample Size')
        ax1.set_xlabel('Age Group × Gender', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(age_gender_stats['age_gender'], rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='black')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, age_gender_stats['count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

        # Line plot for accuracy (red)
        ax2 = ax1.twinx()
        line = ax2.plot(x_pos, age_gender_stats['accuracy'] * 100,
                       color='red', marker='o', linewidth=2.5, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 100])

        # Add accuracy labels on line
        for i, (x, acc) in enumerate(zip(x_pos, age_gender_stats['accuracy'])):
            ax2.text(x, acc * 100 + 2, f'{acc*100:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

        # Title and legend
        plt.title('Accuracy of Error Detection by Age Group and Gender',
                 fontsize=14, fontweight='bold', pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'age_gender_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: age_gender_analysis.png")

    def plot_cross_demographic_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot cross-demographic analysis (age x gender)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Filter valid predictions (keep NA)
        valid_df = self.filter_valid_data(df, exclude_na_errors=False)
        age_order = ['Pediatric (0-17)', 'Young Adult (18-35)', 'Middle-aged (36-55)', 'Older Adult (56+)']

        # 1. Heatmap: Accuracy by Age Group and Gender
        cross_acc = valid_df.groupby(['age_group', 'gender'])['is_correct'].mean().unstack(fill_value=0)
        cross_acc = cross_acc.reindex([ag for ag in age_order if ag in cross_acc.index])

        sns.heatmap(cross_acc, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                   ax=axes[0, 0], cbar_kws={'label': 'Accuracy'})
        axes[0, 0].set_title('Accuracy: Age Group × Gender', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Gender')
        axes[0, 0].set_ylabel('Age Group')

        # 2. Heatmap: Case Count by Age Group and Gender
        cross_count = pd.crosstab(df['age_group'], df['gender'])
        cross_count = cross_count.reindex([ag for ag in age_order if ag in cross_count.index])

        sns.heatmap(cross_count, annot=True, fmt='d', cmap='Blues',
                   ax=axes[0, 1], cbar_kws={'label': 'Case Count'})
        axes[0, 1].set_title('Case Distribution: Age Group × Gender', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Gender')
        axes[0, 1].set_ylabel('Age Group')

        # 3. Grouped bar chart: Accuracy by Age and Gender
        age_gender_acc = valid_df.groupby(['age_group', 'gender'])['is_correct'].mean().unstack()
        age_gender_acc = age_gender_acc.reindex([ag for ag in age_order if ag in age_gender_acc.index])
        age_gender_acc.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'hotpink', 'gray'])
        axes[1, 0].set_title('Accuracy by Age Group and Gender', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Age Group')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend(title='Gender', loc='lower right')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Specialty performance by gender
        specialty_gender_acc = valid_df.groupby(['specialty', 'gender'])['is_correct'].mean().unstack()
        specialty_gender_acc.plot(kind='barh', ax=axes[1, 1], color=['steelblue', 'hotpink', 'gray'])
        axes[1, 1].set_title('Specialty Accuracy by Gender', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Specialty')
        axes[1, 1].legend(title='Gender', loc='lower right')
        axes[1, 1].set_xlim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'cross_demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: cross_demographic_analysis.png")

    def plot_error_type_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot accuracy of error detection across error types with bar+line format."""
        # Filter valid predictions (keep NA)
        valid_df = self.filter_valid_data(df, exclude_na_errors=False)

        # Consolidate error types <5% into Others
        valid_df_consolidated = self.consolidate_others(valid_df, 'error_type', threshold=0.05)

        # Calculate metrics by error type
        error_stats = valid_df_consolidated.groupby('error_type').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        error_stats.columns = ['error_type', 'accuracy', 'count']

        # Sort by accuracy descending
        error_stats = error_stats.sort_values('accuracy', ascending=False)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Bar chart for sample size (muted sky blue for regular, grey for Others)
        x_pos = range(len(error_stats))
        colors = ['#87CEEB' if et != 'Others' else '#808080' for et in error_stats['error_type']]
        bars = ax1.bar(x_pos, error_stats['count'], color=colors, alpha=0.7, label='Sample Size')
        ax1.set_xlabel('Error Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(error_stats['error_type'], rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='black')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, error_stats['count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

        # Line plot for accuracy (red)
        ax2 = ax1.twinx()
        line = ax2.plot(x_pos, error_stats['accuracy'] * 100,
                       color='red', marker='o', linewidth=2.5, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 100])

        # Add accuracy labels on line
        for i, (x, acc) in enumerate(zip(x_pos, error_stats['accuracy'])):
            ax2.text(x, acc * 100 + 2, f'{acc*100:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

        # Title and legend
        plt.title('Accuracy of Error Detection Across Error Types',
                 fontsize=14, fontweight='bold', pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'error_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: error_type_analysis.png")

    def plot_data_distributions(self, df: pd.DataFrame, output_dir: Path):
        """Plot data distributions for error types, age/gender, and labels."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Error Type Distribution
        error_dist = df['error_type'].value_counts().sort_values(ascending=False)
        axes[0, 0].bar(range(len(error_dist)), error_dist.values, color='#87CEEB', alpha=0.8)
        axes[0, 0].set_title('Distribution of Error Types', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Error Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(error_dist)))
        axes[0, 0].set_xticklabels(error_dist.index, rotation=45, ha='right')
        for i, v in enumerate(error_dist.values):
            axes[0, 0].text(i, v + max(error_dist.values)*0.01, str(v), ha='center', fontsize=9)

        # 2. Age Group and Gender Distribution
        age_gender_dist = df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)
        age_order = ['Pediatric (0-17)', 'Young Adult (18-35)', 'Middle-aged (36-55)', 'Older Adult (56+)', 'Unknown']
        age_gender_dist = age_gender_dist.reindex([ag for ag in age_order if ag in age_gender_dist.index])

        age_gender_dist.plot(kind='bar', stacked=False, ax=axes[0, 1], color=['#87CEEB', '#FFB6C1', '#D3D3D3'], alpha=0.8)
        axes[0, 1].set_title('Distribution of Age Groups and Gender', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Gender', loc='upper right')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Label Distribution (Ground Truth)
        label_dist = df['ground_truth'].value_counts()
        label_names = ['Correct' if l == 0 else 'Incorrect' for l in label_dist.index]
        axes[1, 0].bar(label_names, label_dist.values, color=['#87CEEB', "#853131"], alpha=0.8)
        axes[1, 0].set_title('Distribution of Ground Truth Labels', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Count')
        for i, v in enumerate(label_dist.values):
            axes[1, 0].text(i, v + max(label_dist.values)*0.01, str(v), ha='center', fontsize=10, fontweight='bold')

        # 4. Age Distribution Histogram
        ages = df['age'].dropna()
        axes[1, 1].hist(ages, bins=20, color='#87CEEB', alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('Distribution of Patient Ages', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Age (years)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
        axes[1, 1].axvline(ages.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {ages.median():.1f}')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'data_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: data_distributions.png")

    def generate_summary_statistics(self, df: pd.DataFrame, output_dir: Path):
        """Generate and save summary statistics."""
        # Filter valid predictions (excluding NA error types)
        valid_df = self.filter_valid_data(df, exclude_na_errors=True)

        summary = {
            'Overall Statistics': {
                'Total Cases': len(df),
                'Cases with Errors (NA excluded)': len(valid_df),
                'NA Cases Excluded': len(df) - len(valid_df),
                'Overall Accuracy': valid_df['is_correct'].mean(),
                'Mean Confidence': df['confidence_score'].mean() if 'confidence_score' in df else None,
                'Mean Execution Time': df['execution_time'].mean() if 'execution_time' in df else None
            },
            'By Medical Specialty': valid_df.groupby('specialty')['is_correct'].agg(['count', 'mean']).to_dict('index'),
            'By Age Group': valid_df.groupby('age_group')['is_correct'].agg(['count', 'mean']).to_dict('index'),
            'By Gender': valid_df.groupby('gender')['is_correct'].agg(['count', 'mean']).to_dict('index'),
            'By Error Type': valid_df.groupby('error_type')['is_correct'].agg(['count', 'mean']).to_dict('index'),
        }

        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"summary_statistics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved: summary_statistics_{timestamp}.json")

        # Save detailed CSV
        csv_path = output_dir / f"detailed_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: detailed_results_{timestamp}.csv")

    def run_analysis(self, output_dir: str = "evaluation/plots"):
        """Run complete analysis and generate all plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("COMPREHENSIVE ERROR ANALYSIS")
        print("="*70)

        # Load and merge data
        count = self.load_data()
        if count == 0:
            print("No prediction results found.")
            return

        self.merge_data()
        df = self.create_dataframe()

        # Show filtering statistics
        total_cases = len(df)
        na_cases = len(df[df['error_type'].isin(['NA', 'N/A', 'na', 'n/a'])])
        cases_with_errors = total_cases - na_cases

        print(f"\nData Statistics:")
        print(f"  Total cases: {total_cases}")
        print(f"  Cases with NA error type (excluded from analysis): {na_cases}")
        print(f"  Cases with actual errors (analyzed): {cases_with_errors}")

        print(f"\nGenerating plots in: {output_path}")

        # Generate all plots
        print("\n1. Data Distributions...")
        self.plot_data_distributions(df, output_path)

        print("\n2. Medical Specialty Analysis...")
        self.plot_specialty_analysis(df, output_path)

        print("\n3. Patient Population Analysis...")
        self.plot_patient_population_analysis(df, output_path)

        print("\n4. Error Type Analysis...")
        self.plot_error_type_analysis(df, output_path)

        print("\n5. Age × Gender Analysis...")
        self.plot_age_gender_analysis(df, output_path)

        print("\n6. Overall Performance Analysis...")
        self.plot_overall_performance(df, output_path)

        print("\n7. Detailed Demographic Analysis...")
        self.plot_demographic_analysis(df, output_path)

        print("\n8. Cross-Demographic Analysis...")
        self.plot_cross_demographic_analysis(df, output_path)

        print("\n9. Summary Statistics...")
        self.generate_summary_statistics(df, output_path)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print(f"All plots and statistics saved to: {output_path}")
        print("\nGenerated Plots:")
        print("  1. data_distributions.png - Error types, age/gender, labels distribution")
        print("  2. specialty_analysis.png - Accuracy across medical specialties (bar+line)")
        print("  3. patient_population_analysis.png - Accuracy across age groups (bar+line)")
        print("  4. error_type_analysis.png - Accuracy across error types (bar+line)")
        print("  5. age_gender_analysis.png - Accuracy by age×gender (bar+line)")
        print("  6. overall_performance.png - Confusion matrix and metrics")
        print("  7. demographic_analysis.png - Detailed demographic breakdowns")
        print("  8. cross_demographic_analysis.png - Cross-demographic heatmaps")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive error analysis with demographic and specialty breakdowns"
    )
    parser.add_argument(
        "--validation-file", "-v",
        type=str,
        default="test_data/validation.json",
        help="Path to validation.json"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="logs/debates",
        help="Directory containing result_*.json files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation/plots",
        help="Directory to save plots and analysis results"
    )

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = ComprehensiveErrorAnalyzer(
        validation_file=args.validation_file,
        results_dir=args.results_dir
    )

    analyzer.run_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
