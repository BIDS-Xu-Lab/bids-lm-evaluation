#!/usr/bin/env python3
"""
EHRShot Results Extraction and Summarization Script

This script extracts metrics from lm-evaluation-harness results for EHRShot tasks
and provides organized summaries in CSV format. Automatically deduplicates results
to keep only the most recent evaluation for each model/task/context combination.

Supports both old metrics (acc,none, f1,none) and new metrics (exact_match,binary_yn, 
f1_gu_yn,binary_yn, recall_gu_yn,binary_yn).

Usage:
    python extract_results.py [--results_dir PATH] [--output_dir PATH] [--level LEVEL]
    
    --level options:
        group      : Include only group-level aggregated results (default)
        individual : Include only individual task results
"""

import json
import os
import sys
import glob
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import re
# from datetime import datetime  # No longer needed since we're not using timestamps
from typing import Dict, List, Any, Optional

class EHRShotResultsExtractor:
    """Extract and summarize EHRShot evaluation results from lm-evaluation-harness."""
    
    def __init__(self, results_dir: str = None, output_dir: str = None):
        """
        Initialize the results extractor.
        
        Args:
            results_dir: Path to the results directory (default: auto-detect)
            output_dir: Path to save extracted results (default: same as script location)
        """
        # Auto-detect results directory if not provided
        if results_dir is None:
            script_dir = Path(__file__).parent
            # Look for results in common locations relative to results_process
            possible_paths = [
                script_dir / "../../../results/ehr_llm/ehrshot",
                Path("/gpfs/radev/home/yl2342/project/bids-lm-evaluation/results/ehr_llm/ehrshot"),
                Path("/gpfs/radev/home/yl2342/project/lm-evaluation-harness/results/ehr_llm/ehrshot"),
                script_dir.parent.parent.parent / "results/ehr_llm/ehrshot",
                Path("./results/ehr_llm/ehrshot"),
            ]
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    results_dir = str(path.resolve())
                    break
            
            if results_dir is None:
                raise ValueError("Could not auto-detect results directory. Please provide --results_dir")
        
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "extracted_results"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize data structures
        self.all_results = []
        self.summary_data = defaultdict(lambda: defaultdict(dict))
        
    def extract_all_results(self) -> List[Dict[str, Any]]:
        """Extract results from all JSON files in the results directory."""
        
        # Find all result JSON files
        pattern = str(self.results_dir / "**" / "results_*.json")
        result_files = glob.glob(pattern, recursive=True)
        
        print(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                result_data = self._extract_single_result(file_path)
                if result_data:
                    self.all_results.append(result_data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
        print(f"Successfully extracted {len(self.all_results)} results")
        return self.all_results
    
    def _extract_single_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract data from a single result JSON file."""
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Parse file path to extract metadata
        path_parts = Path(file_path).relative_to(self.results_dir).parts
        
        # Extract structure: task_type/[subtask_type/]max_len_X/model_name/results_timestamp.json
        if len(path_parts) < 4:
            print(f"Unexpected path structure: {file_path}")
            return None
            
        task_type = path_parts[0]  # e.g., task_diagnosis, task_mortality
        
        # Handle different path structures
        if path_parts[1].startswith('max_len_'):
            # Direct structure: task_type/max_len_X/model_name/results_file.json
            max_len = path_parts[1]
            model_name = path_parts[2]
            subtask_type = None
        else:
            # Nested structure: task_type/subtask_type/max_len_X/model_name/results_file.json
            subtask_type = path_parts[1]  # e.g., new, recurrent, labs, vitals
            max_len = path_parts[2]
            model_name = path_parts[3]
        
        # Extract timestamp from filename
        filename = path_parts[-1]
        timestamp_match = re.search(r'results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json', filename)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # Extract model configuration
        model_config = data.get('config', {})
        
        # Extract main results
        results = data.get('results', {})
        groups = data.get('groups', {})
        
        extracted_data = {
            'file_path': file_path,
            'task_type': task_type,
            'subtask_type': subtask_type,
            'max_len': max_len,
            'model_name': model_name,
            'timestamp': timestamp,
            'model_config': model_config,
            'results': results,
            'groups': groups,
            'n_samples': data.get('n-samples', {}),
            'versions': data.get('versions', {}),
        }
        
        return extracted_data
    
    def _clean_task_name(self, task_name: str) -> str:
        """Clean task names to remove unnecessary prefixes."""
        if not task_name:
            return task_name
        
        # Remove ehrshot_ prefix
        cleaned = task_name.replace('ehrshot_', '', 1)  # Only remove the first occurrence
        
        return cleaned

    def _clean_parent_group_name(self, parent_group: str) -> str:
        """Clean parent group names to remove unnecessary prefixes and suffixes."""
        if not parent_group or parent_group == 'unknown_group':
            return parent_group
        
        # Remove common prefixes and suffixes
        cleaned = (parent_group
            .replace('group_ehrshot_', '')
            .replace('ehrshot_', '')
            .replace('_group', '')
            .replace('_tasks', '')
            .replace('group_', ''))
        
        # Map specific variations to standardized names
        if 'inpatient' in cleaned.lower():
            return 'inpatient_related'
        elif 'mortality' in cleaned.lower():
            return 'mortality'
        elif 'operational' in cleaned.lower():
            return 'operational'
        elif 'lab' in cleaned.lower() or 'laboratory' in cleaned.lower():
            return 'measurement_lab'
        elif 'vital' in cleaned.lower():
            return 'measurement_vital'
        elif 'new' in cleaned.lower() and 'diagnosis' in cleaned.lower():
            return 'new_diagnosis'
        elif 'recurrent' in cleaned.lower() and 'diagnosis' in cleaned.lower():
            return 'recurrent_diagnosis'
        elif cleaned.lower() in ['new_diagnosis', 'recurrent_diagnosis', 'measurement_lab', 'measurement_vital', 'operational', 'mortality', 'inpatient_related']:
            return cleaned.lower()
        else:
            # For any other cases, return the cleaned version
            return cleaned

    def _get_task_group_mapping(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Create a mapping from individual tasks to their parent groups."""
        task_to_group = {}
        
        # Get group subtasks mapping from the results if available
        group_subtasks = result.get('group_subtasks', {})
        
        for group_name, subtask_list in group_subtasks.items():
            for subtask in subtask_list:
                # Clean the group name before storing
                cleaned_group = self._clean_parent_group_name(group_name)
                task_to_group[subtask] = cleaned_group
        
        # Enhanced pattern-based mapping for the 6 main parent groups
        # This ensures we capture all tasks even if group_subtasks is incomplete
        for task_name in result.get('results', {}):
            if task_name not in task_to_group and task_name not in result.get('groups', {}):
                # Determine parent group based on task name patterns
                if 'mortality' in task_name.lower():
                    task_to_group[task_name] = 'mortality'
                elif 'operational' in task_name.lower():
                    task_to_group[task_name] = 'operational'
                elif 'inpatient' in task_name.lower():
                    task_to_group[task_name] = 'inpatient_related'
                elif 'lab' in task_name.lower() or 'laboratory' in task_name.lower():
                    task_to_group[task_name] = 'measurement_lab'
                elif 'vital' in task_name.lower() or 'vitals' in task_name.lower():
                    task_to_group[task_name] = 'measurement_vital'
                elif 'new' in task_name.lower() and 'diagnosis' in task_name.lower():
                    task_to_group[task_name] = 'new_diagnosis'
                elif 'recurrent' in task_name.lower() and 'diagnosis' in task_name.lower():
                    task_to_group[task_name] = 'recurrent_diagnosis'
                # Additional patterns based on common EHRShot task naming conventions
                elif task_name.startswith('ehrshot_') and 'diagnosis' in task_name:
                    # Try to infer from task structure
                    if 'new' in task_name or task_name.endswith('_new'):
                        task_to_group[task_name] = 'new_diagnosis'
                    elif 'recurrent' in task_name or task_name.endswith('_recurrent'):
                        task_to_group[task_name] = 'recurrent_diagnosis'
                    else:
                        # Default to new diagnosis if unclear
                        task_to_group[task_name] = 'new_diagnosis'
                elif task_name.startswith('ehrshot_') and ('measurement' in task_name or 'lab' in task_name):
                    task_to_group[task_name] = 'measurement_lab'
                elif task_name.startswith('ehrshot_') and 'vital' in task_name:
                    task_to_group[task_name] = 'measurement_vital'
                else:
                    # For any remaining unmapped tasks, try to infer from context
                    # Check if it matches any known group patterns in the groups section
                    groups = result.get('groups', {})
                    matched = False
                    for group_name in groups:
                        if any(keyword in task_name.lower() for keyword in [
                            'mortality', 'operational', 'inpatient', 'lab', 'vital', 'diagnosis'
                        ]):
                            if 'mortality' in group_name.lower() and 'mortality' in task_name.lower():
                                task_to_group[task_name] = 'mortality'
                                matched = True
                                break
                            elif 'operational' in group_name.lower() and 'operational' in task_name.lower():
                                task_to_group[task_name] = 'operational'
                                matched = True
                                break
                            elif 'inpatient' in group_name.lower() and 'inpatient' in task_name.lower():
                                task_to_group[task_name] = 'inpatient_related'
                                matched = True
                                break
                            elif 'lab' in group_name.lower() and 'lab' in task_name.lower():
                                task_to_group[task_name] = 'measurement_lab'
                                matched = True
                                break
                            elif 'vital' in group_name.lower() and 'vital' in task_name.lower():
                                task_to_group[task_name] = 'measurement_vital'
                                matched = True
                                break
                            elif 'diagnosis' in group_name.lower() and 'diagnosis' in task_name.lower():
                                if 'new' in group_name.lower() and ('new' in task_name.lower() or task_name.endswith('_new')):
                                    task_to_group[task_name] = 'new_diagnosis'
                                    matched = True
                                    break
                                elif 'recurrent' in group_name.lower() and ('recurrent' in task_name.lower() or task_name.endswith('_recurrent')):
                                    task_to_group[task_name] = 'recurrent_diagnosis'
                                    matched = True
                                    break
                    
                    if not matched:
                        task_to_group[task_name] = 'unknown_group'
        
        return task_to_group

    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame with key metrics for all results."""
        
        summary_rows = []
        
        for result in self.all_results:
            base_info = {
                'file_path': result['file_path'],
                'task_type': result['task_type'],
                'subtask_type': result['subtask_type'],
                'max_len': result['max_len'],
                'model_name': result['model_name'],
                'timestamp': result['timestamp'],
            }
            
            # Get task to group mapping for this result
            task_to_group = self._get_task_group_mapping(result)
            
            # Extract group-level metrics (summary metrics)
            for group_name, group_metrics in result['groups'].items():
                # Clean the group name for consistency
                cleaned_group_name = self._clean_parent_group_name(group_name)
                
                row = base_info.copy()
                row.update({
                    'task_name': group_name,
                    'task_level': 'group',
                    'parent_group': cleaned_group_name,  # Use cleaned name
                    'accuracy': group_metrics.get('exact_match,binary_yn', group_metrics.get('acc,none')),
                    'accuracy_stderr': group_metrics.get('exact_match_stderr,binary_yn', group_metrics.get('acc_stderr,none')),
                    'f1_score': group_metrics.get('f1_gu_yn,binary_yn', group_metrics.get('f1,none')),
                    'f1_stderr': group_metrics.get('f1_gu_yn_stderr,binary_yn', group_metrics.get('f1_stderr,none')),
                    'recall_score': group_metrics.get('recall_gu_yn,binary_yn'),
                    'recall_stderr': group_metrics.get('recall_gu_yn_stderr,binary_yn'),
                })
                summary_rows.append(row)
            
            # Extract individual task metrics
            for task_name, task_metrics in result['results'].items():
                if task_name in result['groups']:
                    continue  # Skip group entries in results
                    
                # Get parent group for this task
                parent_group = task_to_group.get(task_name, 'unknown_group')
                
                # Clean the task name
                cleaned_task_name = self._clean_task_name(task_name)
                
                row = base_info.copy()
                row.update({
                    'task_name': cleaned_task_name,  # Use cleaned task name
                    'task_level': 'individual',
                    'parent_group': parent_group,
                    'accuracy': task_metrics.get('exact_match,binary_yn', task_metrics.get('acc,none')),
                    'accuracy_stderr': task_metrics.get('exact_match_stderr,binary_yn', task_metrics.get('acc_stderr,none')),
                    'f1_score': task_metrics.get('f1_gu_yn,binary_yn', task_metrics.get('f1,none')),
                    'f1_stderr': task_metrics.get('f1_gu_yn_stderr,binary_yn', task_metrics.get('f1_stderr,none')),
                    'recall_score': task_metrics.get('recall_gu_yn,binary_yn'),
                    'recall_stderr': task_metrics.get('recall_gu_yn_stderr,binary_yn'),
                })
                
                # Add sample count if available
                if task_name in result['n_samples']:
                    row['n_samples'] = result['n_samples'][task_name].get('effective', 
                                                                         result['n_samples'][task_name].get('original'))
                
                summary_rows.append(row)
                
                # Special handling for tasks that are both individual and should be treated as groups
                # (like ehrshot_mortality which doesn't have a separate group-level result)
                if task_name == 'ehrshot_mortality':
                    # Create a pseudo-group entry for mortality
                    group_row = row.copy()
                    group_row['task_level'] = 'group'
                    group_row['task_name'] = cleaned_task_name  # Use cleaned name for group entry too
                    group_row['parent_group'] = 'mortality'  # Use cleaned name
                    summary_rows.append(group_row)
        
        return pd.DataFrame(summary_rows)
    
    def deduplicate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate results for the same model, task, and context size.
        Keeps the most recent result based on timestamp.
        """
        # Sort by timestamp to get most recent first
        df_sorted = df.sort_values('timestamp', ascending=False)
        
        # Define deduplication keys
        dedup_keys = ['model_name', 'max_len', 'task_name', 'task_level']
        
        # Track unique JSON files before and after deduplication
        unique_files_before = df['file_path'].nunique()
        
        # Remove duplicates, keeping the first (most recent) occurrence
        df_dedup = df_sorted.drop_duplicates(subset=dedup_keys, keep='first')
        
        # Track unique JSON files after deduplication
        unique_files_after = df_dedup['file_path'].nunique()
        
        # Count files dropped
        files_dropped = unique_files_before - unique_files_after
        result_entries_removed = len(df) - len(df_dedup)
        
        if files_dropped > 0:
            print(f"Dropped {files_dropped} duplicate JSON files (kept {unique_files_after} most recent files)")
            print(f"This removed {result_entries_removed} duplicate result entries")
        
        return df_dedup
    
    def generate_summary_report(self, df: pd.DataFrame, level_filter: str = 'group') -> str:
        """Generate a text summary report of the results."""
        
        # First deduplicate the data
        df_clean = self.deduplicate_results(df)
        
        # Filter by task level
        if level_filter == 'group':
            df_clean = df_clean[df_clean['task_level'] == 'group']
            report_title = "EHRShot Evaluation Results Summary (Group Level)"
        elif level_filter == 'individual':
            df_clean = df_clean[df_clean['task_level'] == 'individual']
            report_title = "EHRShot Evaluation Results Summary (Individual Tasks)"
        else:
            raise ValueError(f"Invalid level_filter: {level_filter}. Must be 'group' or 'individual'")
        
        report = [report_title, "=" * len(report_title), ""]
        
        # Basic statistics
        total_experiments = len(df)
        total_after_dedup = len(df_clean)
        unique_models = df_clean['model_name'].nunique()
        unique_tasks = df_clean['task_name'].nunique()
        
        report.extend([
            f"Total experiments (raw): {total_experiments}",
            f"Total experiments (after deduplication): {total_after_dedup}",
            f"Unique models: {unique_models}",
            f"Unique tasks: {unique_tasks}",
            ""
        ])
        
        # Model performance overview
        report.extend(["Model Performance Overview:", "-" * 30])
        
        # Handle different levels
        if level_filter == 'group':
            # Group-level performance
            group_df = df_clean[df_clean['task_level'] == 'group'].copy()
            if not group_df.empty:
                # First show group-specific performance
                columns_to_show = ['model_name', 'max_len', 'task_name', 'accuracy', 'f1_score']
                if 'recall_score' in group_df.columns:
                    columns_to_show.append('recall_score')
                group_specific = group_df[columns_to_show].copy()
                group_specific = group_specific.sort_values(['task_name', 'model_name', 'max_len'])
                
                report.append("\nGroup-specific Performance:")
                report.append(group_specific.to_string(index=False))
                
                # Then show aggregated performance across all groups
                agg_metrics = {'accuracy': 'mean', 'f1_score': 'mean'}
                if 'recall_score' in group_df.columns:
                    agg_metrics['recall_score'] = 'mean'
                model_performance = group_df.groupby(['model_name', 'max_len']).agg(agg_metrics).round(4)
                
                report.append("\n\nAggregated Performance (Average across all groups):")
                report.append(model_performance.to_string())
        
        elif level_filter == 'individual':
            # Individual-level performance
            individual_df = df_clean[df_clean['task_level'] == 'individual'].copy()
            if not individual_df.empty:
                # First show individual task-specific performance with parent group
                columns_to_show = ['model_name', 'max_len', 'parent_group', 'task_name', 'accuracy', 'f1_score']
                if 'recall_score' in individual_df.columns:
                    columns_to_show.append('recall_score')
                individual_specific = individual_df[columns_to_show].copy()
                individual_specific = individual_specific.sort_values(['parent_group', 'task_name', 'model_name', 'max_len'])
                
                report.append("\nIndividual Task Performance:")
                report.append(individual_specific.to_string(index=False))
        
        # Task type breakdown
        report.extend(["\n\nTask Type Breakdown:", "-" * 20])
        
        # Handle cases where subtask_type might be None/empty
        df_copy = df_clean.copy()
        df_copy['subtask_type'] = df_copy['subtask_type'].fillna('none')
        
        agg_dict = {
            'task_name': 'nunique',
            'accuracy': 'mean',
            'f1_score': 'mean'
        }
        if 'recall_score' in df_copy.columns:
            agg_dict['recall_score'] = 'mean'
        
        task_breakdown = df_copy.groupby(['task_type', 'subtask_type']).agg(agg_dict).round(4)
        
        col_names = ['num_tasks', 'avg_accuracy', 'avg_f1']
        if 'recall_score' in df_copy.columns:
            col_names.append('avg_recall')
        task_breakdown.columns = col_names
        
        report.append(task_breakdown.to_string())
        
        # Best performing models per task type
        report.extend(["\n\nBest Performing Models by Task Type:", "-" * 40])
        
        for task_type in df_clean['task_type'].unique():
            if pd.isna(task_type):
                continue
                
            # First try group-level results
            task_df = df_clean[(df_clean['task_type'] == task_type) & (df_clean['task_level'] == 'group')]
            
            # If no group-level results, use individual-level results
            if task_df.empty:
                task_df = df_clean[(df_clean['task_type'] == task_type) & (df_clean['task_level'] == 'individual')]
                level_note = " (individual tasks)"
            else:
                level_note = ""
            
            if not task_df.empty:
                best_acc = task_df.loc[task_df['accuracy'].idxmax()]
                best_f1 = task_df.loc[task_df['f1_score'].idxmax()]
                
                report_lines = [
                    f"\n{task_type}{level_note}:",
                    f"  Best Accuracy: {best_acc['model_name']} ({best_acc['max_len']}) - {best_acc['accuracy']:.4f}",
                    f"  Best F1: {best_f1['model_name']} ({best_f1['max_len']}) - {best_f1['f1_score']:.4f}"
                ]
                
                if 'recall_score' in task_df.columns and not task_df['recall_score'].isna().all():
                    best_recall = task_df.loc[task_df['recall_score'].idxmax()]
                    report_lines.append(f"  Best Recall: {best_recall['model_name']} ({best_recall['max_len']}) - {best_recall['recall_score']:.4f}")
                
                report.extend(report_lines)
        
        return "\n".join(report)
    
    def create_comparison_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create comparison tables for different views of the data."""
        
        tables = {}
        
        # Handle cases where subtask_type might be None/empty
        df_copy = df.copy()
        df_copy['subtask_type'] = df_copy['subtask_type'].fillna('none')
        
        # Model comparison table (group-level metrics only)
        group_df = df_copy[df_copy['task_level'] == 'group'].copy()
        if not group_df.empty:
            values_list = ['accuracy', 'f1_score']
            if 'recall_score' in group_df.columns:
                values_list.append('recall_score')
            
            model_comparison = group_df.pivot_table(
                index=['model_name', 'max_len'],
                columns=['task_type', 'subtask_type'],
                values=values_list,
                aggfunc='first'
            )
            tables['model_comparison'] = model_comparison
        
        # Task type summary
        agg_dict = {
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std', 'count']
        }
        if 'recall_score' in df_copy.columns:
            agg_dict['recall_score'] = ['mean', 'std', 'count']
            
        task_summary = df_copy.groupby(['task_type', 'subtask_type', 'task_level']).agg(agg_dict).round(4)
        tables['task_summary'] = task_summary
        
        # Max length impact analysis
        if 'max_len' in df.columns:
            agg_dict = {
                'accuracy': 'mean',
                'f1_score': 'mean'
            }
            if 'recall_score' in df.columns:
                agg_dict['recall_score'] = 'mean'
                
            max_len_analysis = df.groupby(['max_len', 'task_type']).agg(agg_dict).round(4)
            tables['max_len_analysis'] = max_len_analysis
        
        return tables
    
    def save_results(self, df: pd.DataFrame, level_filter: str = 'group'):
        """Save deduplicated results in CSV format."""
        
        # Deduplicate data
        df_clean = self.deduplicate_results(df)
        
        # Filter by task level
        if level_filter == 'group':
            df_clean = df_clean[df_clean['task_level'] == 'group']
            level_suffix = "_group"
        elif level_filter == 'individual':
            df_clean = df_clean[df_clean['task_level'] == 'individual']
            level_suffix = "_individual"
        else:
            raise ValueError(f"Invalid level_filter: {level_filter}. Must be 'group' or 'individual'")
        
        # Remove file_path column before saving (used only for deduplication tracking)
        df_to_save = df_clean.drop(columns=['file_path'], errors='ignore')
        
        # Save CSV file (overwrite latest results)
        output_file = self.output_dir / f"ehrshot_results_summary{level_suffix}.csv"
        df_to_save.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Save summary report
        summary_report = self.generate_summary_report(df, level_filter)
        report_file = self.output_dir / f"ehrshot_summary_report{level_suffix}.txt"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        print(f"Summary report saved to: {report_file}")
        
        return summary_report

def main():
    """Main function to run the results extraction."""
    
    parser = argparse.ArgumentParser(description='Extract and summarize EHRShot evaluation results')
    parser.add_argument('--results_dir', type=str, help='Path to results directory')
    parser.add_argument('--output_dir', type=str, help='Path to save extracted results')
    parser.add_argument('--level', type=str, choices=['group', 'individual'], 
                       default='group', help='Filter results by task level (default: group)')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = EHRShotResultsExtractor(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        
        # Extract all results
        print("Extracting results from JSON files...")
        extractor.extract_all_results()
        
        # Create summary DataFrame
        print("Creating summary DataFrame...")
        df = extractor.create_summary_dataframe()
        
        if df.empty:
            print("No results found!")
            return
        
        print(f"Created summary with {len(df)} rows")
        
        # Save results and generate report
        print(f"Saving deduplicated results ({args.level} level) in CSV format...")
        summary_report = extractor.save_results(df, args.level)
        
        # Print summary to console
        print("\n" + summary_report)
        
        print(f"\nExtraction completed successfully!")
        print(f"Results saved to: {extractor.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
