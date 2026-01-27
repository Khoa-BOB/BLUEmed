#!/usr/bin/env python3
"""
Demographic Analysis for Medical Error Detection Evaluation.

Extracts demographics (age, gender) from medical notes and analyzes
prediction performance across different demographic groups.

Usage:
    python evaluation/demographic_analysis.py
    python evaluation/demographic_analysis.py --results-dir logs/debates --output-dir evaluation/analysis
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DemographicAnalyzer:
    """Analyzes prediction results by demographic factors."""

    # Age group definitions
    AGE_GROUPS = {
        'Pediatric (0-17)': (0, 17),
        'Young Adult (18-35)': (18, 35),
        'Middle-aged (36-55)': (36, 55),
        'Older Adult (56+)': (56, 200)
    }

    # Disease type classification keywords
    DISEASE_KEYWORDS = {
        'Infectious Disease': [
            'infection', 'bacterial', 'viral', 'virus', 'bacteria', 'sepsis', 'septic',
            'pneumonia', 'tuberculosis', 'HIV', 'AIDS', 'hepatitis', 'meningitis',
            'gonorrhea', 'chlamydia', 'syphilis', 'herpes', 'influenza', 'flu',
            'streptococcus', 'staphylococcus', 'e. coli', 'salmonella', 'malaria',
            'typhoid', 'cholera', 'UTI', 'urinary tract infection', 'cellulitis',
            'abscess', 'fever', 'antimicrobial', 'antibiotic', 'pathogen', 'organism'
        ],
        'Cardiovascular': [
            'heart', 'cardiac', 'cardiovascular', 'myocardial', 'infarction',
            'arrhythmia', 'atrial', 'ventricular', 'hypertension', 'hypotension',
            'angina', 'coronary', 'aortic', 'valve', 'murmur', 'palpitation',
            'tachycardia', 'bradycardia', 'chest pain', 'EKG', 'ECG', 'echocardiogram'
        ],
        'Respiratory': [
            'lung', 'pulmonary', 'respiratory', 'bronchitis', 'asthma', 'COPD',
            'emphysema', 'cough', 'dyspnea', 'shortness of breath', 'wheeze',
            'stridor', 'pneumothorax', 'pleural', 'bronchial', 'alveolar'
        ],
        'Gastrointestinal': [
            'stomach', 'intestinal', 'bowel', 'colon', 'gastric', 'hepatic', 'liver',
            'pancreatic', 'pancreatitis', 'appendicitis', 'cholecystitis', 'diarrhea',
            'vomiting', 'nausea', 'abdominal pain', 'GI', 'gastrointestinal'
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
            'wound', 'ulcer', 'blister', 'erythema', 'pruritus', 'itching'
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
        ],
        'Immunological': [
            'immune', 'autoimmune', 'allergy', 'allergic', 'immunodeficiency',
            'HIV', 'AIDS', 'lupus', 'rheumatoid'
        ]
    }

    # Department classification keywords
    DEPARTMENT_KEYWORDS = {
        'Emergency Medicine': [
            'emergency department', 'emergency room', 'ER', 'ED', 'trauma',
            'acute', 'urgent', 'brought to', 'comes to the emergency'
        ],
        'Internal Medicine': [
            'internist', 'internal medicine', 'general medicine', 'primary care',
            'comes to the physician', 'comes to the doctor', 'follow-up', 'outpatient'
        ],
        'Pediatrics': [
            'pediatric', 'pediatrician', 'child', 'infant', 'newborn', 'neonate',
            'toddler', 'adolescent', 'year-old boy', 'year-old girl', 'brought by mother',
            'brought by father', 'brought by parent'
        ],
        'Obstetrics/Gynecology': [
            'obstetric', 'gynecology', 'OB/GYN', 'pregnant', 'pregnancy', 'prenatal',
            'postpartum', 'menstrual', 'vaginal', 'cervical', 'uterine', 'ovarian'
        ],
        'Surgery': [
            'surgery', 'surgical', 'operation', 'post-operative', 'pre-operative',
            'incision', 'resection', 'removal', 'arthroplasty', 'appendectomy'
        ],
        'Cardiology': [
            'cardiology', 'cardiologist', 'heart', 'cardiac', 'EKG', 'echocardiogram',
            'chest pain', 'palpitation', 'arrhythmia'
        ],
        'Pulmonology': [
            'pulmonology', 'pulmonologist', 'lung', 'respiratory', 'breathing',
            'asthma', 'COPD', 'bronchitis'
        ],
        'Infectious Disease': [
            'infectious disease', 'infection', 'sepsis', 'HIV', 'AIDS',
            'antimicrobial', 'antibiotic therapy'
        ],
        'Dermatology': [
            'dermatology', 'dermatologist', 'skin', 'rash', 'lesion', 'dermatitis'
        ],
        'Neurology': [
            'neurology', 'neurologist', 'seizure', 'stroke', 'headache', 'numbness'
        ],
        'Orthopedics': [
            'orthopedic', 'orthopedist', 'fracture', 'joint', 'bone', 'arthroplasty'
        ],
        'Urology': [
            'urology', 'urologist', 'kidney', 'bladder', 'urinary', 'prostate'
        ]
    }

    def __init__(
        self,
        validation_file: str = "test_data/validation.json",
        results_dir: str = "logs/debates"
    ):
        """
        Initialize analyzer.

        Args:
            validation_file: Path to validation.json with original test cases
            results_dir: Directory containing result_*.json prediction files
        """
        self.validation_file = Path(validation_file)
        self.results_dir = Path(results_dir)

        # Data storage
        self.validation_data = {}  # id -> case data
        self.results_data = {}     # id -> prediction result
        self.merged_data = []      # Combined records with demographics

    def load_data(self) -> int:
        """Load validation data and prediction results."""
        # Load validation data
        print(f"Loading validation data from {self.validation_file}...")
        with open(self.validation_file, 'r', encoding='utf-8') as f:
            validation_list = json.load(f)

        for case in validation_list:
            case_id = case.get('id')
            if case_id:
                self.validation_data[case_id] = case
        print(f"  Loaded {len(self.validation_data)} validation cases")

        # Load prediction results
        print(f"Loading prediction results from {self.results_dir}...")
        result_files = list(self.results_dir.glob("result_*.json"))

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
        """
        Extract age from medical note text.

        Args:
            text: Medical note text

        Returns:
            Age as integer or None if not found
        """
        # Pattern: "A 24-year-old", "24-year-old", "aged 24", "age 24", "24 years old"
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
                if 0 <= age <= 120:  # Sanity check
                    return age
        return None

    def extract_gender(self, text: str) -> Optional[str]:
        """
        Extract gender from medical note text.

        Args:
            text: Medical note text

        Returns:
            'Male', 'Female', or None if not found
        """
        text_lower = text.lower()

        # Check for explicit gender indicators
        male_patterns = [
            r'\b(man|male|boy|gentleman|he|his|him)\b',
            r'\b(\d+[-\s]?year[-\s]?old)\s+(man|male|boy)\b',
        ]
        female_patterns = [
            r'\b(woman|female|girl|lady|she|her)\b',
            r'\b(\d+[-\s]?year[-\s]?old)\s+(woman|female|girl)\b',
        ]

        # Count matches
        male_count = sum(len(re.findall(p, text_lower)) for p in male_patterns)
        female_count = sum(len(re.findall(p, text_lower)) for p in female_patterns)

        # More specific patterns (higher weight)
        if re.search(r'\d+[-\s]?year[-\s]?old\s+(woman|female|girl)', text_lower):
            return 'Female'
        if re.search(r'\d+[-\s]?year[-\s]?old\s+(man|male|boy)', text_lower):
            return 'Male'

        # Fall back to pronoun analysis
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

    def classify_disease_type(self, text: str) -> str:
        """Classify disease type based on keywords in medical note."""
        text_lower = text.lower()
        scores = {}

        for disease_type, keywords in self.DISEASE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[disease_type] = score

        if scores:
            return max(scores, key=scores.get)
        return 'Other/Unclassified'

    def classify_department(self, text: str, age: Optional[int] = None) -> str:
        """Classify department based on keywords and patient age."""
        text_lower = text.lower()
        scores = {}

        # Check for pediatric age first
        if age is not None and age < 18:
            scores['Pediatrics'] = 10  # High base score for pediatric patients

        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[dept] = scores.get(dept, 0) + score

        if scores:
            return max(scores, key=scores.get)
        return 'General Medicine'

    def merge_data(self) -> List[Dict]:
        """Merge validation data with prediction results and extract demographics."""
        self.merged_data = []

        for case_id, result in self.results_data.items():
            # Get validation case
            val_case = self.validation_data.get(case_id, {})
            text = val_case.get('text', result.get('medical_note', ''))

            # Extract demographics
            age = self.extract_age(text)
            gender = self.extract_gender(text)
            age_group = self.get_age_group(age)

            # Extract disease type and department
            disease_type = self.classify_disease_type(text)
            department = self.classify_department(text, age)

            # Build merged record
            record = {
                'case_id': case_id,

                # Demographics
                'age': age,
                'age_group': age_group,
                'gender': gender if gender else 'Unknown',

                # Disease & Department
                'disease_type': disease_type,
                'department': department,

                # Ground truth
                'ground_truth': result.get('ground_truth', val_case.get('label')),
                'error_type': result.get('error_type', val_case.get('error_type', 'NA')),

                # Predictions
                'predicted_label': result.get('predicted_label'),
                'final_answer': result.get('final_answer'),
                'confidence_score': result.get('confidence_score'),
                'confidence_normalized': result.get('confidence_normalized'),

                # Debate info
                'winner': result.get('winner'),
                'expert_a_source': result.get('expert_a', {}).get('source', 'Unknown'),
                'expert_b_source': result.get('expert_b', {}).get('source', 'Unknown'),

                # Correctness
                'is_correct': result.get('ground_truth') == result.get('predicted_label'),

                # Metadata
                'execution_time': result.get('execution_time'),
                'text_length': len(text)
            }

            self.merged_data.append(record)

        print(f"Merged {len(self.merged_data)} records with demographics")
        return self.merged_data

    def calculate_group_metrics(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for a group of records."""
        if not records:
            return {'count': 0, 'accuracy': None}

        y_true = [r['ground_truth'] for r in records if r['ground_truth'] is not None]
        y_pred = [r['predicted_label'] for r in records if r['predicted_label'] is not None]

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            return {'count': len(records), 'accuracy': None}

        # Basic metrics
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true)

        metrics = {
            'count': len(records),
            'correct': correct,
            'accuracy': accuracy,
        }

        if SKLEARN_AVAILABLE and len(set(y_true)) > 1:
            metrics.update({
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
            })

        # Confidence stats
        confidences = [r['confidence_normalized'] for r in records if r['confidence_normalized'] is not None]
        if confidences:
            metrics['avg_confidence'] = sum(confidences) / len(confidences)
            if NUMPY_AVAILABLE:
                metrics['std_confidence'] = float(np.std(confidences))

        # ROC-AUC calculation using confidence scores
        # For ROC-AUC, we need probability scores for the positive class (INCORRECT=1)
        if SKLEARN_AVAILABLE and len(set(y_true)) > 1:
            try:
                # Get confidence scores aligned with predictions
                y_scores = []
                y_true_for_roc = []
                for r in records:
                    if r['ground_truth'] is not None and r['confidence_normalized'] is not None:
                        y_true_for_roc.append(r['ground_truth'])
                        # Adjust score based on prediction direction
                        # If predicted INCORRECT (1), use confidence as-is
                        # If predicted CORRECT (0), use 1-confidence for probability of INCORRECT
                        if r['predicted_label'] == 1:
                            y_scores.append(r['confidence_normalized'])
                        else:
                            y_scores.append(1 - r['confidence_normalized'])

                if len(y_scores) > 0 and len(set(y_true_for_roc)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true_for_roc, y_scores)
            except Exception as e:
                metrics['roc_auc'] = None
                metrics['roc_auc_error'] = str(e)

        return metrics

    def analyze_by_dimension(self, dimension: str) -> Dict[str, Dict]:
        """
        Analyze results grouped by a specific dimension.

        Args:
            dimension: Field name to group by (e.g., 'age_group', 'gender', 'error_type')

        Returns:
            Dictionary mapping dimension values to metrics
        """
        groups = defaultdict(list)

        for record in self.merged_data:
            key = record.get(dimension, 'Unknown')
            if key is None:
                key = 'Unknown'
            groups[key].append(record)

        results = {}
        for key, records in sorted(groups.items()):
            results[key] = self.calculate_group_metrics(records)

        return results

    def analyze_all(self) -> Dict[str, Any]:
        """Run complete demographic analysis."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(self.merged_data),
            'overall_metrics': self.calculate_group_metrics(self.merged_data),
        }

        # Demographic analyses
        print("\nAnalyzing by demographics...")

        # Age group analysis
        analysis['by_age_group'] = self.analyze_by_dimension('age_group')

        # Gender analysis
        analysis['by_gender'] = self.analyze_by_dimension('gender')

        # Error type analysis
        analysis['by_error_type'] = self.analyze_by_dimension('error_type')

        # Winner analysis
        analysis['by_winner'] = self.analyze_by_dimension('winner')

        # Disease type analysis
        print("Analyzing by disease type...")
        analysis['by_disease_type'] = self.analyze_by_dimension('disease_type')

        # Department analysis
        print("Analyzing by department...")
        analysis['by_department'] = self.analyze_by_dimension('department')

        # Cross-tabulations
        print("Generating cross-tabulations...")
        analysis['crosstab'] = {
            'age_group_x_accuracy': self._crosstab('age_group', 'is_correct'),
            'gender_x_accuracy': self._crosstab('gender', 'is_correct'),
            'error_type_x_accuracy': self._crosstab('error_type', 'is_correct'),
            'winner_x_ground_truth': self._crosstab('winner', 'ground_truth'),
        }

        # Distribution summaries
        analysis['distributions'] = {
            'age': self._get_age_distribution(),
            'gender': self._get_value_counts('gender'),
            'error_type': self._get_value_counts('error_type'),
            'winner': self._get_value_counts('winner'),
            'final_answer': self._get_value_counts('final_answer'),
        }

        return analysis

    def _crosstab(self, dim1: str, dim2: str) -> Dict:
        """Create cross-tabulation of two dimensions."""
        crosstab = defaultdict(lambda: defaultdict(int))

        for record in self.merged_data:
            val1 = str(record.get(dim1, 'Unknown'))
            val2 = str(record.get(dim2, 'Unknown'))
            crosstab[val1][val2] += 1

        return {k: dict(v) for k, v in crosstab.items()}

    def _get_age_distribution(self) -> Dict:
        """Get age statistics."""
        ages = [r['age'] for r in self.merged_data if r['age'] is not None]

        if not ages:
            return {'count': 0}

        dist = {
            'count': len(ages),
            'min': min(ages),
            'max': max(ages),
            'mean': sum(ages) / len(ages),
        }

        if NUMPY_AVAILABLE:
            dist['median'] = float(np.median(ages))
            dist['std'] = float(np.std(ages))

        return dist

    def _get_value_counts(self, field: str) -> Dict[str, int]:
        """Get value counts for a field."""
        counts = defaultdict(int)
        for record in self.merged_data:
            val = record.get(field)
            if val is None:
                val = 'Unknown'
            counts[str(val)] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def save_results(self, analysis: Dict, output_dir: str = "evaluation/analysis") -> Tuple[Path, Path]:
        """Save analysis results to JSON and CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = output_path / f"demographic_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON saved to: {json_path}")

        # Save detailed CSV with all records
        csv_path = output_path / f"demographic_details_{timestamp}.csv"
        if self.merged_data:
            headers = list(self.merged_data[0].keys())
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(','.join(headers) + '\n')
                for record in self.merged_data:
                    values = [str(record.get(h, '')).replace(',', ';') for h in headers]
                    f.write(','.join(values) + '\n')
        print(f"CSV saved to: {csv_path}")

        # Save summary CSV
        summary_path = output_path / f"demographic_summary_{timestamp}.csv"
        self._save_summary_csv(analysis, summary_path)
        print(f"Summary saved to: {summary_path}")

        return json_path, csv_path

    def _save_summary_csv(self, analysis: Dict, filepath: Path):
        """Save summary metrics to CSV."""
        rows = []

        # Add metrics for each dimension
        for dim_name in ['by_age_group', 'by_gender', 'by_error_type', 'by_winner']:
            if dim_name in analysis:
                for group, metrics in analysis[dim_name].items():
                    for metric_name, value in metrics.items():
                        rows.append({
                            'dimension': dim_name.replace('by_', ''),
                            'group': group,
                            'metric': metric_name,
                            'value': value
                        })

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('dimension,group,metric,value\n')
            for row in rows:
                f.write(f"{row['dimension']},{row['group']},{row['metric']},{row['value']}\n")

    def print_summary(self, analysis: Dict):
        """Print formatted summary to console."""
        print("\n" + "=" * 70)
        print("DEMOGRAPHIC ANALYSIS SUMMARY")
        print("=" * 70)

        # Overall
        overall = analysis.get('overall_metrics', {})
        print(f"\nTotal Cases: {analysis.get('total_cases', 0)}")
        print(f"Overall Accuracy: {overall.get('accuracy', 0):.4f}")
        if 'f1_score' in overall:
            print(f"Overall F1 Score: {overall.get('f1_score', 0):.4f}")
        if 'roc_auc' in overall and overall.get('roc_auc') is not None:
            print(f"Overall ROC-AUC:  {overall.get('roc_auc'):.4f}")

        # By Age Group
        print("\n" + "-" * 50)
        print("BY AGE GROUP")
        print("-" * 50)
        print(f"{'Age Group':<25} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
        for group, metrics in analysis.get('by_age_group', {}).items():
            acc = metrics.get('accuracy')
            f1 = metrics.get('f1_score')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10} {f1_str:>8}")

        # By Gender
        print("\n" + "-" * 50)
        print("BY GENDER")
        print("-" * 50)
        print(f"{'Gender':<25} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
        for group, metrics in analysis.get('by_gender', {}).items():
            acc = metrics.get('accuracy')
            f1 = metrics.get('f1_score')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10} {f1_str:>8}")

        # By Error Type
        print("\n" + "-" * 50)
        print("BY ERROR TYPE")
        print("-" * 50)
        print(f"{'Error Type':<25} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
        for group, metrics in analysis.get('by_error_type', {}).items():
            acc = metrics.get('accuracy')
            f1 = metrics.get('f1_score')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10} {f1_str:>8}")

        # By Winner
        print("\n" + "-" * 50)
        print("BY DEBATE WINNER")
        print("-" * 50)
        print(f"{'Winner':<25} {'Count':>8} {'Accuracy':>10}")
        for group, metrics in analysis.get('by_winner', {}).items():
            acc = metrics.get('accuracy')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10}")

        # By Disease Type
        print("\n" + "-" * 50)
        print("BY DISEASE TYPE")
        print("-" * 50)
        print(f"{'Disease Type':<25} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
        for group, metrics in analysis.get('by_disease_type', {}).items():
            acc = metrics.get('accuracy')
            f1 = metrics.get('f1_score')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10} {f1_str:>8}")

        # By Department
        print("\n" + "-" * 50)
        print("BY DEPARTMENT")
        print("-" * 50)
        print(f"{'Department':<25} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
        for group, metrics in analysis.get('by_department', {}).items():
            acc = metrics.get('accuracy')
            f1 = metrics.get('f1_score')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"{group:<25} {metrics.get('count', 0):>8} {acc_str:>10} {f1_str:>8}")

        # Distributions
        print("\n" + "-" * 50)
        print("DISTRIBUTIONS")
        print("-" * 50)

        dists = analysis.get('distributions', {})
        if 'age' in dists and dists['age'].get('count', 0) > 0:
            age_dist = dists['age']
            print(f"Age: min={age_dist.get('min')}, max={age_dist.get('max')}, "
                  f"mean={age_dist.get('mean', 0):.1f}, median={age_dist.get('median', 'N/A')}")

        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demographic analysis of prediction results")
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
        default="evaluation/analysis",
        help="Directory to save analysis results"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DemographicAnalyzer(
        validation_file=args.validation_file,
        results_dir=args.results_dir
    )

    # Load data
    count = analyzer.load_data()
    if count == 0:
        print("No prediction results found. Run batch_predict.py first.")
        return

    # Merge and extract demographics
    analyzer.merge_data()

    # Run analysis
    analysis = analyzer.analyze_all()

    # Print summary
    analyzer.print_summary(analysis)

    # Save results
    analyzer.save_results(analysis, output_dir=args.output_dir)

    print("\nDemographic analysis complete!")


if __name__ == "__main__":
    main()
