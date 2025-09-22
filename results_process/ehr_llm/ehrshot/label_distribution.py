#!/usr/bin/env python3
"""
Label Distribution Analysis for EHRShot Tasks

This script analyzes the binary label distribution (0/1) across all EHRShot datasets.
Based on analysis of utils.py files, it identifies the correct outcome variable for each task type
and analyzes representative datasets from each group. Supports different data splits (test, total, etc.).
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Any
import sys
import json


class EHRShotLabelAnalyzer:
    """Analyzes label distribution in EHRShot datasets based on task type outcome variables."""
    
    def __init__(self, data_base_path: str = None, split_suffix: str = "_test"):
        """
        Initialize the label analyzer.
        
        Args:
            data_base_path: Base path to the ehrshot data. If None, uses default path.
            split_suffix: Suffix for the data split to analyze (e.g., "_test", "_all")
        """
        if data_base_path is None:
            self.data_base_path = "/home/yl2342/project_pi_hx235/yl2342/data/ehrshot/visit_oriented_ehr"
        else:
            self.data_base_path = data_base_path
            
        if not os.path.exists(self.data_base_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_base_path}")
        
        self.split_suffix = split_suffix
        
        # Define task groups and their outcome variables based on utils.py analysis
        self.task_groups = {
            'Mortality': {
                'data_path': f'task_mortality/task_mortality{split_suffix}',
                'outcome_column': 'mortality_30d',
                'description': '30-day inpatient mortality prediction (0=survived, 1=deceased)'
            },
            'Operational_Readmission': {
                'data_path': f'task_operational/task_readmission/task_readmission{split_suffix}',
                'outcome_column': 'readmission_30d',
                'description': '30-day readmission prediction (0=no readmission, 1=readmitted)'
            },
            'Operational_Longstay': {
                'data_path': f'task_operational/task_longstay/task_longstay{split_suffix}',
                'outcome_column': 'longstay_7d',
                'description': '7-day long stay prediction (0=short stay, 1=long stay)'
            },
            'New_Diagnosis': {
                'data_path': f'task_diagnosis/task_diagnosis{split_suffix}',
                'outcome_column': 'hypertension_one_year_diagnosis',  # Using hypertension as representative
                'description': 'New diagnosis prediction within 1 year (0=no diagnosis, 1=diagnosis) - hypertension representative'
            },
            'Recurrent_Diagnosis': {
                'data_path': f'task_diagnosis/task_diagnosis{split_suffix}',
                'outcome_column': 'hypertension_one_year_diagnosis',  # Using hypertension as representative
                'description': 'Recurrent diagnosis prediction within 1 year (0=no diagnosis, 1=diagnosis) - hypertension representative'
            },
            'Lab_Measurements': {
                'data_path': f'task_measurement/glucose{split_suffix}',
                'outcome_column': 'next_label',
                'description': 'Lab measurement prediction (0=normal, 1=abnormal) - glucose representative'
            },
            'Vital_Measurements': {
                'data_path': f'task_measurement/sbp{split_suffix}',
                'outcome_column': 'next_label',
                'description': 'Vital measurement prediction (0=normal, 1=abnormal) - SBP representative'
            }
        }
    
    def load_parquet_data(self, data_path: str) -> pd.DataFrame:
        """Load parquet data from the specified path pattern."""
        full_path = os.path.join(self.data_base_path, data_path)
        
        try:
            # Handle directory with multiple parquet files
            if os.path.isdir(full_path):
                parquet_files = glob.glob(os.path.join(full_path, "*.parquet"))
                if not parquet_files:
                    print(f"No parquet files found in: {full_path}")
                    return None
                
                # Load and concatenate all files
                dataframes = []
                for file_path in parquet_files:
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
                
                if dataframes:
                    return pd.concat(dataframes, ignore_index=True)
                else:
                    return None
            else:
                print(f"Directory not found: {full_path}")
                return None
                    
        except Exception as e:
            print(f"Error loading data from {full_path}: {e}")
            return None
    
    def analyze_task_labels(self, task_name: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze label distribution for a specific task group.
        
        Args:
            task_name: Name of the task group
            task_info: Dictionary with data_path, outcome_column, and description
            
        Returns:
            Dictionary with label distribution statistics
        """
        print(f"\nAnalyzing {task_name}...")
        print(f"Data path: {task_info['data_path']}")
        print(f"Outcome column: {task_info['outcome_column']}")
        
        # Load data
        df = self.load_parquet_data(task_info['data_path'])
        if df is None or df.empty:
            print(f"No data found for {task_name}")
            return None
            
        print(f"Loaded {len(df)} samples")
        
        # Check if outcome column exists
        outcome_col = task_info['outcome_column']
        if outcome_col not in df.columns:
            print(f"Warning: Outcome column '{outcome_col}' not found in {task_name}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
            
        labels = df[outcome_col]
        
        # Handle different data types
        if labels.dtype == 'bool':
            labels = labels.astype(int)
        elif labels.dtype == 'object':
            # Try to convert string labels to numeric
            try:
                labels = pd.to_numeric(labels)
            except:
                print(f"Warning: Could not convert labels to numeric for {task_name}")
                return None
        
        # Calculate statistics
        unique_labels = sorted(labels.dropna().unique())
        total_samples = len(labels.dropna())
        
        if total_samples == 0:
            print(f"No valid labels found for {task_name}")
            return None
        
        distribution = {}
        for label in unique_labels:
            count = (labels == label).sum()
            percentage = (count / total_samples) * 100
            distribution[f'label_{int(label)}'] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Calculate class balance metrics for binary classification
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            positive_count = (labels == 1).sum()
            negative_count = (labels == 0).sum()
            
            imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
            minority_class_pct = min(positive_count, negative_count) / total_samples * 100
            
            balance_metrics = {
                'imbalance_ratio': round(imbalance_ratio, 2),
                'minority_class_percentage': round(minority_class_pct, 2),
                'positive_percentage': round((positive_count / total_samples) * 100, 2),
                'negative_percentage': round((negative_count / total_samples) * 100, 2),
                'is_balanced': imbalance_ratio <= 1.5  # Consider balanced if ratio <= 1.5
            }
        else:
            balance_metrics = {
                'note': f'Non-binary classification detected with labels: {unique_labels}'
            }
        
        return {
            'task_name': task_name,
            'data_path': task_info['data_path'],
            'outcome_column': outcome_col,
            'description': task_info['description'],
            'total_samples': total_samples,
            'unique_labels': [int(x) for x in unique_labels],
            'distribution': distribution,
            'balance_metrics': balance_metrics
        }
    
    def analyze_all_diagnosis_tasks(self) -> Dict[str, Any]:
        """
        Analyze ALL diagnosis tasks individually for both new and recurrent cases.
        """
        diagnosis_results = {}
        
        # Load diagnosis data once
        df = self.load_parquet_data(f'task_diagnosis/task_diagnosis{self.split_suffix}')
        if df is None or df.empty:
            print("No diagnosis data found")
            return {}
            
        # Find all diagnosis outcome columns
        diagnosis_cols = [col for col in df.columns if col.endswith('_one_year_diagnosis')]
        
        print(f"\nAnalyzing ALL {len(diagnosis_cols)} diagnosis conditions...")
        
        # Analyze each condition for new vs recurrent
        for diag_col in diagnosis_cols:
            condition_name = diag_col.replace('_one_year_diagnosis', '')
            type_col = f"{condition_name}_type"
            
            if type_col in df.columns:
                # Analyze new cases
                new_cases = df[df[type_col] == 'new']
                if len(new_cases) > 0:
                    labels = new_cases[diag_col].dropna()
                    if len(labels) > 0:
                        total = len(labels)
                        positive = int((labels == 1).sum())
                        negative = total - positive
                        positive_rate = round((positive / total) * 100, 2)
                        
                        task_name = f"ehrshot_new_diagnosis_{condition_name}"
                        diagnosis_results[task_name] = {
                            'task_name': task_name,
                            'condition': condition_name.replace('_', ' ').title(),
                            'task_type': 'New Diagnosis',
                            'total_count': total,
                            'positive_count': positive,
                            'negative_count': negative,
                            'positive_rate': positive_rate
                        }
                        print(f"  {task_name}: {total:,} total, {positive:,} positive ({positive_rate:.1f}%)")
                
                # Analyze recurrent cases
                recurrent_cases = df[df[type_col] == 'recurrent']
                if len(recurrent_cases) > 0:
                    labels = recurrent_cases[diag_col].dropna()
                    if len(labels) > 0:
                        total = len(labels)
                        positive = int((labels == 1).sum())
                        negative = total - positive
                        positive_rate = round((positive / total) * 100, 2)
                        
                        task_name = f"ehrshot_recurrent_diagnosis_{condition_name}"
                        diagnosis_results[task_name] = {
                            'task_name': task_name,
                            'condition': condition_name.replace('_', ' ').title(),
                            'task_type': 'Recurrent Diagnosis',
                            'total_count': total,
                            'positive_count': positive,
                            'negative_count': negative,
                            'positive_rate': positive_rate
                        }
                        print(f"  {task_name}: {total:,} total, {positive:,} positive ({positive_rate:.1f}%)")
        
        return diagnosis_results
    
    def analyze_all_measurement_tasks(self) -> Dict[str, Any]:
        """
        Analyze ALL measurement tasks individually (labs and vitals).
        """
        measurement_results = {}
        
        # Define all measurement types to analyze
        measurement_types = {
            # Lab measurements
            'glucose': (f'task_measurement/glucose{self.split_suffix}', 'Lab'),
            'creatinine': (f'task_measurement/creatinine{self.split_suffix}', 'Lab'),
            'hemoglobin': (f'task_measurement/hemoglobin{self.split_suffix}', 'Lab'),
            'hba1c': (f'task_measurement/hba1c{self.split_suffix}', 'Lab'),
            'albumin': (f'task_measurement/albumin{self.split_suffix}', 'Lab'),
            'alt': (f'task_measurement/alt{self.split_suffix}', 'Lab'),
            'ast': (f'task_measurement/ast{self.split_suffix}', 'Lab'),
            'bilirubin': (f'task_measurement/bilirubin{self.split_suffix}', 'Lab'),
            'bun': (f'task_measurement/bun{self.split_suffix}', 'Lab'),
            'crp': (f'task_measurement/crp{self.split_suffix}', 'Lab'),
            'ldl': (f'task_measurement/ldl{self.split_suffix}', 'Lab'),
            'platelets': (f'task_measurement/platelets{self.split_suffix}', 'Lab'),
            'totalcholesterol': (f'task_measurement/totalcholesterol{self.split_suffix}', 'Lab'),
            # Vital measurements
            'sbp': (f'task_measurement/sbp{self.split_suffix}', 'Vital'),
            'dbp': (f'task_measurement/dbp{self.split_suffix}', 'Vital'),
            'heart_rate': (f'task_measurement/heart_rate{self.split_suffix}', 'Vital')
        }
        
        print(f"\nAnalyzing ALL {len(measurement_types)} measurement tasks...")
        
        for measurement_name, (data_path, measurement_type) in measurement_types.items():
            df = self.load_parquet_data(data_path)
            if df is not None and 'next_label' in df.columns:
                labels = df['next_label'].dropna()
                if len(labels) > 0:
                    total = len(labels)
                    positive = int((labels == 1).sum())
                    negative = total - positive
                    positive_rate = round((positive / total) * 100, 2)
                    
                    task_name = f"ehrshot_measurement_{measurement_type.lower()}_{measurement_name}"
                    measurement_results[task_name] = {
                        'task_name': task_name,
                        'measurement': measurement_name.replace('_', ' ').title(),
                        'task_type': f'{measurement_type} Measurement',
                        'total_count': total,
                        'positive_count': positive,
                        'negative_count': negative,
                        'positive_rate': positive_rate
                    }
                    print(f"  {task_name}: {total:,} total, {positive:,} abnormal ({positive_rate:.1f}%)")
        
        return measurement_results
    
    def analyze_individual_operational_tasks(self) -> Dict[str, Any]:
        """Analyze individual operational tasks."""
        operational_results = {}
        
        # Mortality task
        print(f"\nAnalyzing Mortality task...")
        df = self.load_parquet_data(f'task_mortality/task_mortality{self.split_suffix}')
        if df is not None and 'mortality_30d' in df.columns:
            labels = df['mortality_30d'].dropna()
            if len(labels) > 0:
                total = len(labels)
                positive = int((labels == 1).sum())
                negative = total - positive
                positive_rate = round((positive / total) * 100, 2)
                
                task_name = "ehrshot_mortality"
                operational_results[task_name] = {
                    'task_name': task_name,
                    'description': '30-day mortality prediction',
                    'task_type': 'Operational',
                    'total_count': total,
                    'positive_count': positive,
                    'negative_count': negative,
                    'positive_rate': positive_rate
                }
                print(f"  {task_name}: {total:,} total, {positive:,} deceased ({positive_rate:.1f}%)")
        
        # Readmission task
        print(f"\nAnalyzing Readmission task...")
        df = self.load_parquet_data(f'task_operational/task_readmission/task_readmission{self.split_suffix}')
        if df is not None and 'readmission_30d' in df.columns:
            labels = df['readmission_30d'].dropna()
            if len(labels) > 0:
                total = len(labels)
                positive = int((labels == 1).sum())
                negative = total - positive
                positive_rate = round((positive / total) * 100, 2)
                
                task_name = "ehrshot_operational_readmission"
                operational_results[task_name] = {
                    'task_name': task_name,
                    'description': '30-day readmission prediction',
                    'task_type': 'Operational',
                    'total_count': total,
                    'positive_count': positive,
                    'negative_count': negative,
                    'positive_rate': positive_rate
                }
                print(f"  {task_name}: {total:,} total, {positive:,} readmitted ({positive_rate:.1f}%)")
        
        # Long stay task
        print(f"\nAnalyzing Long stay task...")
        df = self.load_parquet_data(f'task_operational/task_longstay/task_longstay{self.split_suffix}')
        if df is not None and 'longstay_7d' in df.columns:
            labels = df['longstay_7d'].dropna()
            if len(labels) > 0:
                total = len(labels)
                positive = int((labels == 1).sum())
                negative = total - positive
                positive_rate = round((positive / total) * 100, 2)
                
                task_name = "ehrshot_operational_longstay"
                operational_results[task_name] = {
                    'task_name': task_name,
                    'description': '7-day long stay prediction',
                    'task_type': 'Operational',
                    'total_count': total,
                    'positive_count': positive,
                    'negative_count': negative,
                    'positive_rate': positive_rate
                }
                print(f"  {task_name}: {total:,} total, {positive:,} long stay ({positive_rate:.1f}%)")
        
        return operational_results

    def analyze_all_tasks(self) -> Dict[str, Any]:
        """Analyze label distribution for all ehrshot tasks individually."""
        print("Starting EHRShot label distribution analysis...")
        print(f"Data directory: {self.data_base_path}")
        
        results = {}
        
        # Analyze individual operational tasks
        operational_tasks = self.analyze_individual_operational_tasks()
        if operational_tasks:
            results.update(operational_tasks)
        
        # Analyze all individual diagnosis tasks
        diagnosis_tasks = self.analyze_all_diagnosis_tasks()
        if diagnosis_tasks:
            results.update(diagnosis_tasks)
        
        # Analyze all individual measurement tasks
        measurement_tasks = self.analyze_all_measurement_tasks()
        if measurement_tasks:
            results.update(measurement_tasks)
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary report with markdown tables."""
        report = []
        report.append("# EHRShot Label Distribution Analysis Report")
        report.append("")
        
        # Filter tasks by type
        operational_tasks = {k: v for k, v in results.items() if 'task_type' in v and v['task_type'] == 'Operational'}
        new_diagnosis_tasks = {k: v for k, v in results.items() if 'task_type' in v and v['task_type'] == 'New Diagnosis'}
        recurrent_diagnosis_tasks = {k: v for k, v in results.items() if 'task_type' in v and v['task_type'] == 'Recurrent Diagnosis'}
        lab_measurement_tasks = {k: v for k, v in results.items() if 'task_type' in v and v['task_type'] == 'Lab Measurement'}
        vital_measurement_tasks = {k: v for k, v in results.items() if 'task_type' in v and v['task_type'] == 'Vital Measurement'}
        
        total_tasks = len(operational_tasks) + len(new_diagnosis_tasks) + len(recurrent_diagnosis_tasks) + len(lab_measurement_tasks) + len(vital_measurement_tasks)
        
        report.append(f"**Total Individual Tasks Analyzed: {total_tasks}**")
        report.append("")
        report.append(f"- Operational Tasks: {len(operational_tasks)}")
        report.append(f"- New Diagnosis Tasks: {len(new_diagnosis_tasks)}")
        report.append(f"- Recurrent Diagnosis Tasks: {len(recurrent_diagnosis_tasks)}")
        report.append(f"- Lab Measurement Tasks: {len(lab_measurement_tasks)}")
        report.append(f"- Vital Measurement Tasks: {len(vital_measurement_tasks)}")
        report.append("")
        
        # Helper function to create markdown table
        def create_task_table(task_dict, task_type_name):
            if not task_dict:
                return []
            
            table_lines = []
            table_lines.append(f"## {task_type_name}")
            table_lines.append("")
            table_lines.append("| Task Name | Total Count | Positive Count | Negative Count | Positive Rate (%) |")
            table_lines.append("|-----------|-------------|----------------|----------------|-------------------|")
            
            # Sort by positive rate descending
            sorted_tasks = sorted(task_dict.items(), key=lambda x: x[1]['positive_rate'], reverse=True)
            
            for task_name, stats in sorted_tasks:
                table_lines.append(f"| {stats['task_name']} | {stats['total_count']:,} | {stats['positive_count']:,} | {stats['negative_count']:,} | {stats['positive_rate']:.2f}% |")
            
            table_lines.append("")
            
            # Summary statistics
            total_samples = sum(task['total_count'] for task in task_dict.values())
            total_positive = sum(task['positive_count'] for task in task_dict.values())
            avg_positive_rate = sum(task['positive_rate'] for task in task_dict.values()) / len(task_dict)
            
            table_lines.append(f"**{task_type_name} Summary:**")
            table_lines.append(f"- Total samples: {total_samples:,}")
            table_lines.append(f"- Total positive cases: {total_positive:,}")
            table_lines.append(f"- Average positive rate: {avg_positive_rate:.2f}%")
            table_lines.append("")
            
            return table_lines
        
        # Add each task type section
        report.extend(create_task_table(operational_tasks, "Operational Tasks"))
        report.extend(create_task_table(new_diagnosis_tasks, "New Diagnosis Tasks"))
        report.extend(create_task_table(recurrent_diagnosis_tasks, "Recurrent Diagnosis Tasks"))
        report.extend(create_task_table(lab_measurement_tasks, "Lab Measurement Tasks"))
        report.extend(create_task_table(vital_measurement_tasks, "Vital Measurement Tasks"))
        
        # Overall summary
        report.append("## Overall Summary")
        report.append("")
        
        all_individual_tasks = {k: v for k, v in results.items() if 'task_type' in v}
        total_samples_all = sum(task['total_count'] for task in all_individual_tasks.values())
        total_positive_all = sum(task['positive_count'] for task in all_individual_tasks.values())
        overall_positive_rate = (total_positive_all / total_samples_all) * 100 if total_samples_all > 0 else 0
        
        report.append(f"- **Total tasks analyzed:** {len(all_individual_tasks)}")
        report.append(f"- **Total samples across all tasks:** {total_samples_all:,}")
        report.append(f"- **Total positive cases:** {total_positive_all:,}")
        report.append(f"- **Overall positive rate:** {overall_positive_rate:.2f}%")
        report.append("")
        
        # Distribution statistics by task type
        report.append("### Positive Rate Distribution by Task Type")
        report.append("")
        
        for task_type_name, task_dict in [
            ("Operational", operational_tasks),
            ("New Diagnosis", new_diagnosis_tasks), 
            ("Recurrent Diagnosis", recurrent_diagnosis_tasks),
            ("Lab Measurement", lab_measurement_tasks),
            ("Vital Measurement", vital_measurement_tasks)
        ]:
            if task_dict:
                rates = [task['positive_rate'] for task in task_dict.values()]
                min_rate = min(rates)
                max_rate = max(rates)
                avg_rate = sum(rates) / len(rates)
                
                report.append(f"**{task_type_name}:**")
                report.append(f"- Range: {min_rate:.2f}% - {max_rate:.2f}%")
                report.append(f"- Average: {avg_rate:.2f}%")
                report.append("")
        
        # Most and least balanced tasks
        if all_individual_tasks:
            sorted_by_rate = sorted(all_individual_tasks.items(), key=lambda x: x[1]['positive_rate'])
            most_balanced = [task for task in sorted_by_rate if 40 <= task[1]['positive_rate'] <= 60]
            most_imbalanced = [task for task in sorted_by_rate if task[1]['positive_rate'] < 5 or task[1]['positive_rate'] > 95]
            
            if most_balanced:
                report.append("### Most Balanced Tasks (40-60% positive rate)")
                report.append("")
                for task_name, stats in most_balanced[:5]:  # Show top 5
                    report.append(f"- {stats['task_name']}: {stats['positive_rate']:.2f}%")
                report.append("")
            
            if most_imbalanced:
                report.append("### Most Imbalanced Tasks (<5% or >95% positive rate)")
                report.append("")
                for task_name, stats in most_imbalanced[:5]:  # Show top 5
                    report.append(f"- {stats['task_name']}: {stats['positive_rate']:.2f}%")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None):
        """Save results to files."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "label_distribution_analysis"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        with open(output_dir / "label_distribution_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary CSV for individual tasks
        summary_data = []
        for task_name, analysis in results.items():
            if isinstance(analysis, dict) and 'task_type' in analysis:
                row = {
                    'task_name': analysis['task_name'],
                    'task_type': analysis['task_type'],
                    'total_count': analysis['total_count'],
                    'positive_count': analysis['positive_count'],
                    'negative_count': analysis['negative_count'],
                    'positive_rate': analysis['positive_rate']
                }
                
                # Add additional fields if available
                if 'condition' in analysis:
                    row['condition'] = analysis['condition']
                if 'measurement' in analysis:
                    row['measurement'] = analysis['measurement']
                if 'description' in analysis:
                    row['description'] = analysis['description']
                    
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            # Sort by task type and positive rate
            summary_df = summary_df.sort_values(['task_type', 'positive_rate'], ascending=[True, False])
            summary_df.to_csv(output_dir / "label_distribution_summary.csv", index=False)
        
        # Save summary report
        report = self.generate_summary_report(results)
        with open(output_dir / "label_distribution_report.md", 'w') as f:
            f.write(report)
            
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Detailed JSON: label_distribution_detailed.json")
        print(f"  - Summary CSV: label_distribution_summary.csv")
        print(f"  - Markdown Report: label_distribution_report.md")


def main():
    """Main function to run the label distribution analysis."""
    parser = argparse.ArgumentParser(description="Analyze label distribution in EHRShot datasets")
    parser.add_argument("--data_path", type=str, help="Path to ehrshot data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--split", type=str, choices=["test", "total"], default="test",
                       help="Data split to analyze. Choices: 'test', 'total'. Default: 'test'")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        # Map split choices to actual directory suffixes
        split_mapping = {"test": "_test", "total": "_total"}
        split_suffix = split_mapping[args.split]
        analyzer = EHRShotLabelAnalyzer(data_base_path=args.data_path, split_suffix=split_suffix)
        
        # Run analysis
        results = analyzer.analyze_all_tasks()
        
        if not results:
            print("No results found. Please check that the data files exist and are accessible.")
            return
        
        # Generate and display summary
        report = analyzer.generate_summary_report(results)
        print("\n" + report)
        
        # Save results
        analyzer.save_results(results, args.output_dir)
        
        print(f"\nAnalysis complete! Processed {len(results)} task groups.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 