#!/usr/bin/env python3
"""
EHRShot Results Visualization Script

This script creates comprehensive visualizations from the extracted EHRShot evaluation results.
Supports both group-level and individual-level task analysis with enhanced parent group stratification.

Features:
    - Task type/parent group heatmaps
    - Max context length analysis
    - Parent group performance analysis
    - Individual task breakdown within groups
    - Radar plots for model comparison across group tasks
    - Comprehensive performance overview dashboard

Usage:
    python visualize_results.py [--data_file PATH] [--output_dir PATH] [--level LEVEL]
    
    --level: Filter by task level (group, individual). Required parameter.
    
Generated Visualizations:
    - max_length_analysis.png (shows accuracy, F1, and recall across different context lengths)
    - task_type_heatmap.png (shows accuracy, F1, and recall across task types and models)
    - radar_plot_combined_max_len_*.png (model comparison for accuracy, F1, and recall by context length)
    - parent_group_analysis_{accuracy|f1_score}.png (individual level only)
    - individual_task_breakdown_{accuracy|f1_score}.png (individual level only)
    
Examples:
    # Visualize only group-level results
    python visualize_results.py --level group
    
    # Visualize individual tasks with parent group analysis
    python visualize_results.py --level individual
    
    # Use specific data file and output directory
    python visualize_results.py --data_file extracted_results/ehrshot_results_summary_individual.csv --output_dir my_plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
# from datetime import datetime  # No longer needed since we removed timestamps
from typing import List, Optional

class EHRShotResultsVisualizer:
    """Create visualizations for EHRShot evaluation results."""
    
    def __init__(self, output_dir: str = None, level: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Path to save visualization files
            level: Task level (group or individual) to determine subdirectory
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(__file__).parent / "visualizations"
            if level == 'group':
                self.output_dir = base_dir / "group_tasks"
            elif level == 'individual':
                self.output_dir = base_dir / "individual_tasks"
            else:
                self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load data from CSV file."""
        if data_file.endswith('.csv'):
            return pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            return pd.read_json(data_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    

    
    def create_task_type_heatmap(self, df: pd.DataFrame) -> Optional[str]:
        """Create a combined heatmap showing accuracy, F1, and recall performance across task types and models."""
        
        # Filter for group-level results
        group_df = df[df['task_level'] == 'group'].copy()
        
        if group_df.empty:
            print("No group-level data available for task type heatmap")
            return None
        
        # Clean up labels
        group_df_clean = group_df.copy()
        if 'parent_group' in group_df.columns:
            # Parent groups are already clean from extraction, just format for display
            group_df_clean['parent_group_clean'] = group_df_clean['parent_group'].str.replace('_', ' ').str.title()
        group_df_clean['model_name_clean'] = group_df_clean['model_name'].str.replace('Qwen__', '', regex=False).str.replace('meta-llama__', '', regex=False)
        
        # Create pivot tables for all three metrics
        if 'parent_group' in group_df.columns:
            accuracy_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='accuracy',
                aggfunc='mean'
            )
            f1_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='f1_score',
                aggfunc='mean'
            )
            recall_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='recall_score',
                aggfunc='mean'
            ) if 'recall_score' in group_df_clean.columns else pd.DataFrame()
        else:
            accuracy_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns=['task_type', 'subtask_type'],
                values='accuracy',
                aggfunc='mean'
            )
            f1_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns=['task_type', 'subtask_type'],
                values='f1_score',
                aggfunc='mean'
            )
            recall_data = group_df_clean.pivot_table(
                index='model_name_clean',
                columns=['task_type', 'subtask_type'],
                values='recall_score',
                aggfunc='mean'
            ) if 'recall_score' in group_df_clean.columns else pd.DataFrame()
        
        if accuracy_data.empty and f1_data.empty and recall_data.empty:
            return None
        
        # Calculate shared scale
        all_values = []
        if not accuracy_data.empty:
            all_values.extend(accuracy_data.values.flatten())
        if not f1_data.empty:
            all_values.extend(f1_data.values.flatten())
        if not recall_data.empty:
            all_values.extend(recall_data.values.flatten())
        
        # Remove NaN values for scale calculation
        all_values = [v for v in all_values if not pd.isna(v)]
        if all_values:
            vmin = max(0, min(all_values) - 0.05)
            vmax = min(1, max(all_values) + 0.05)
        else:
            vmin, vmax = 0, 1
        
        # Create three side-by-side heatmaps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Use a more visually friendly colormap
        cmap = 'viridis'  # More colorblind-friendly and visually appealing
        
        # Accuracy heatmap
        if not accuracy_data.empty:
            sns.heatmap(
                accuracy_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                cbar=False  # We'll add a shared colorbar later
            )
            ax1.set_title('Accuracy')
            ax1.set_xlabel('Task Type')
            ax1.set_ylabel('Model')
            # Rotate x-axis labels to save space
            ax1.tick_params(axis='x', rotation=20)
        
        # F1 Score heatmap
        if not f1_data.empty:
            sns.heatmap(
                f1_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                cbar=False  # We'll add a shared colorbar later
            )
            ax2.set_title('F1')
            ax2.set_xlabel('Task Type')
            ax2.set_ylabel('')  # Remove y-label for middle plot
            # Rotate x-axis labels to save space
            ax2.tick_params(axis='x', rotation=20)
        
        # Recall heatmap
        if not recall_data.empty:
            im = sns.heatmap(
                recall_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                ax=ax3,
                vmin=vmin,
                vmax=vmax,
                cbar=False  # We'll add a shared colorbar later
            )
            ax3.set_title('Recall')
            ax3.set_xlabel('Task Type')
            ax3.set_ylabel('')  # Remove y-label for right plot
            # Rotate x-axis labels to save space
            ax3.tick_params(axis='x', rotation=20)
        
        # Add shared colorbar at the very bottom, below x-axis labels
        if not accuracy_data.empty or not f1_data.empty or not recall_data.empty:
            # Adjust subplot positioning to make more room for colorbar
            plt.subplots_adjust(bottom=0.18)
            
            # Create colorbar positioned at the very bottom, no label to save space
            cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.015])  # [left, bottom, width, height]
            # Use the last available heatmap object for colorbar
            if not recall_data.empty:
                fig.colorbar(im.collections[0], cax=cbar_ax, orientation='horizontal')
            elif not f1_data.empty:
                # Get the mappable from the F1 plot
                for ax_child in ax2.get_children():
                    if hasattr(ax_child, 'collections') and ax_child.collections:
                        fig.colorbar(ax_child.collections[0], cax=cbar_ax, orientation='horizontal')
                        break
            elif not accuracy_data.empty:
                # Get the mappable from the accuracy plot
                for ax_child in ax1.get_children():
                    if hasattr(ax_child, 'collections') and ax_child.collections:
                        fig.colorbar(ax_child.collections[0], cax=cbar_ax, orientation='horizontal')
                        break
        
        plt.suptitle('Task Performance Heatmap', fontsize=16, y=0.98)
        
        # Save plot (overwrite latest)
        output_file = self.output_dir / "task_type_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined task type heatmap saved: {output_file}")
        return str(output_file)
    
    def create_max_length_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create plots showing the impact of max_length on accuracy, F1, and recall performance."""
        
        # Filter for group-level results
        group_df = df[df['task_level'] == 'group'].copy()
        
        if group_df.empty or 'max_len' not in group_df.columns:
            return None
        
        # Extract numeric max_len values
        group_df['max_len_num'] = group_df['max_len'].str.extract(r'(\d+)').astype(int)
        
        # Filter to only include models with multiple max_len values
        models_with_multiple_lengths = []
        for model in group_df['model_name'].unique():
            model_data = group_df[group_df['model_name'] == model]
            unique_lengths = model_data['max_len_num'].nunique()
            if unique_lengths > 1:
                models_with_multiple_lengths.append(model)
        
        if not models_with_multiple_lengths:
            print("No models found with multiple max_len values")
            return None
        
        # Filter data to only include models with multiple lengths
        group_df = group_df[group_df['model_name'].isin(models_with_multiple_lengths)]
        
        # Create figure with subplots for accuracy, F1, and recall
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Calculate combined min/max for all metrics to set shared scale
        all_values = []
        
        for model in models_with_multiple_lengths:
            model_data = group_df[group_df['model_name'] == model]
            if 'accuracy' in model_data.columns:
                acc_values = model_data.groupby('max_len_num')['accuracy'].mean().values
                all_values.extend(acc_values)
            if 'f1_score' in model_data.columns:
                f1_values = model_data.groupby('max_len_num')['f1_score'].mean().values
                all_values.extend(f1_values)
            if 'recall_score' in model_data.columns:
                recall_values = model_data.groupby('max_len_num')['recall_score'].mean().values
                all_values.extend(recall_values)
        
        # Calculate shared y-axis limits
        if all_values:
            y_min = max(0, min(all_values) - 0.05)  # Don't go below 0
            y_max = min(1, max(all_values) + 0.05)   # Don't go above 1
        else:
            y_min, y_max = 0, 1
        
        # Plot 1: Accuracy vs Max Length for each model
        for model in models_with_multiple_lengths:
            model_data = group_df[group_df['model_name'] == model]
            if 'accuracy' in model_data.columns:
                # Clean up model name for legend
                clean_model = model.replace('Qwen__', '').replace('meta-llama__', '')
                length_performance = model_data.groupby('max_len_num')['accuracy'].mean()
                ax1.plot(length_performance.index, length_performance.values, marker='o', label=clean_model)
        
        ax1.set_xlabel('Max Length')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Max Length')
        ax1.set_ylim(y_min, y_max)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 vs Max Length for each model
        for model in models_with_multiple_lengths:
            model_data = group_df[group_df['model_name'] == model]
            if 'f1_score' in model_data.columns:
                # Clean up model name for legend
                clean_model = model.replace('Qwen__', '').replace('meta-llama__', '')
                length_performance = model_data.groupby('max_len_num')['f1_score'].mean()
                ax2.plot(length_performance.index, length_performance.values, marker='s', label=clean_model)
        
        ax2.set_xlabel('Max Length')
        ax2.set_ylabel('F1')
        ax2.set_title('F1 vs Max Length')
        ax2.set_ylim(y_min, y_max)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recall vs Max Length for each model
        for model in models_with_multiple_lengths:
            model_data = group_df[group_df['model_name'] == model]
            if 'recall_score' in model_data.columns:
                # Clean up model name for legend
                clean_model = model.replace('Qwen__', '').replace('meta-llama__', '')
                length_performance = model_data.groupby('max_len_num')['recall_score'].mean()
                ax3.plot(length_performance.index, length_performance.values, marker='^', label=clean_model)
        
        ax3.set_xlabel('Max Length')
        ax3.set_ylabel('Recall')
        ax3.set_title('Recall vs Max Length')
        ax3.set_ylim(y_min, y_max)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot (overwrite latest)
        output_file = self.output_dir / "max_length_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Max length analysis saved: {output_file}")
        return str(output_file)
    
    def create_parent_group_analysis(self, df: pd.DataFrame, metric: str = 'accuracy') -> Optional[str]:
        """Create analysis of performance by parent group for individual tasks."""
        
        # Filter for individual-level results
        individual_df = df[df['task_level'] == 'individual'].copy()
        
        if individual_df.empty or 'parent_group' not in individual_df.columns:
            print("No individual-level data with parent_group column available")
            return None
        
        # Create figure with subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Box plot by parent group
        groups = individual_df['parent_group'].unique()
        group_data = []
        group_labels = []
        
        for group in sorted(groups):
            group_subset = individual_df[individual_df['parent_group'] == group][metric].dropna()
            if not group_subset.empty:
                group_data.append(group_subset)
                group_labels.append(group)
        
        if group_data:
            # Clean up group labels for display
            clean_group_labels = [label.replace('group_ehrshot_', '').replace('_group', '').replace('_tasks', '') for label in group_labels]
            ax1.boxplot(group_data, labels=clean_group_labels)
            ax1.set_ylabel(metric.capitalize())
            ax1.set_title(f'Performance by Parent Group - {metric.capitalize()}')
            ax1.tick_params(axis='x', rotation=20)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average performance by parent group and model
        group_model_avg = individual_df.groupby(['parent_group', 'model_name'])[metric].mean().reset_index()
        
        # Clean up labels for heatmap
        group_model_avg['parent_group_clean'] = group_model_avg['parent_group'].str.replace('group_ehrshot_', '', regex=False).str.replace('_group', '', regex=False).str.replace('_tasks', '', regex=False)
        group_model_avg['model_name_clean'] = group_model_avg['model_name'].str.replace('Qwen__', '', regex=False).str.replace('meta-llama__', '', regex=False)
        
        # Create pivot for heatmap
        pivot_data = group_model_avg.pivot(index='model_name_clean', columns='parent_group_clean', values=metric)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            ax=ax2,
            cbar_kws={'label': metric.capitalize()}
        )
        ax2.set_title(f'Model Performance by Parent Group - {metric.capitalize()}')
        ax2.set_xlabel('Parent Group')
        ax2.set_ylabel('Model')
        
        plt.tight_layout()
        
        # Save plot (overwrite latest)
        output_file = self.output_dir / f"parent_group_analysis_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parent group analysis saved: {output_file}")
        return str(output_file)

    def create_individual_task_breakdown(self, df: pd.DataFrame, metric: str = 'accuracy') -> Optional[str]:
        """Create detailed breakdown of individual task performance within parent groups."""
        
        # Filter for individual-level results
        individual_df = df[df['task_level'] == 'individual'].copy()
        
        if individual_df.empty or 'parent_group' not in individual_df.columns:
            print("No individual-level data with parent_group column available")
            return None
        
        # Get unique parent groups
        parent_groups = individual_df['parent_group'].unique()
        
        # Create subplots - one for each parent group (max 6 to fit on screen)
        n_groups = min(len(parent_groups), 6)
        if n_groups == 0:
            return None
            
        cols = 3 if n_groups > 4 else 2
        rows = (n_groups + cols - 1) // cols
        
        _, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_groups == 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
        else:
            axes = axes.flatten()
        
        for i, parent_group in enumerate(sorted(parent_groups)[:n_groups]):
            ax = axes[i]
            
            # Clean up parent group name for display
            clean_parent_group = parent_group.replace('group_ehrshot_', '').replace('_group', '').replace('_tasks', '')
            
            # Get data for this parent group
            group_data = individual_df[individual_df['parent_group'] == parent_group]
            
            if not group_data.empty:
                # Create bar plot of performance by task within this group
                task_performance = group_data.groupby('task_name')[metric].mean().sort_values(ascending=False)
                
                if len(task_performance) > 0:
                    task_performance.plot(kind='bar', ax=ax)
                    ax.set_title(f'{clean_parent_group}\n({len(task_performance)} tasks)')
                    ax.set_ylabel(metric.capitalize())
                    ax.tick_params(axis='x', rotation=20)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{clean_parent_group}\n(No data)')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{clean_parent_group}\n(No data)')
        
        # Hide unused subplots
        for i in range(n_groups, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot (overwrite latest)
        output_file = self.output_dir / f"individual_task_breakdown_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual task breakdown saved: {output_file}")
        return str(output_file)

    def create_radar_plot(self, df: pd.DataFrame) -> List[str]:
        """Create radar plots comparing models across group tasks for accuracy, F1, and recall, separated by max_len."""
        
        # Filter for group-level results
        group_df = df[df['task_level'] == 'group'].copy()
        
        if group_df.empty or 'parent_group' not in group_df.columns:
            print("No group-level data with parent_group column available for radar plot")
            return []
        
        # Clean up labels
        group_df_clean = group_df.copy()
        # Parent groups are already clean from extraction, just format for display
        group_df_clean['parent_group_clean'] = group_df_clean['parent_group'].str.replace('_', ' ').str.title()
        group_df_clean['model_name_clean'] = group_df_clean['model_name'].str.replace('Qwen__', '', regex=False).str.replace('meta-llama__', '', regex=False)
        
        # Get unique max_len values
        max_lens = sorted(group_df_clean['max_len'].unique())
        saved_files = []
        
        for max_len in max_lens:
            # Filter data for this specific max_len
            max_len_data = group_df_clean[group_df_clean['max_len'] == max_len].copy()
            
            if max_len_data.empty:
                continue
            
            # Create pivot tables for all three metrics
            accuracy_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='accuracy',
                aggfunc='mean'
            )
            
            f1_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='f1_score',
                aggfunc='mean'
            )
            
            recall_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='parent_group_clean',
                values='recall_score',
                aggfunc='mean'
            ) if 'recall_score' in max_len_data.columns else pd.DataFrame()
            
            if accuracy_data.empty and f1_data.empty and recall_data.empty:
                continue
            
            # Get the task groups (dimensions) - use whichever is available
            categories = list(accuracy_data.columns) if not accuracy_data.empty else (
                list(f1_data.columns) if not f1_data.empty else list(recall_data.columns)
            )
            N = len(categories)
            
            if N < 3:
                print(f"Need at least 3 task groups for radar plot (max_len: {max_len}), found {N}")
                continue
            
            # Create angles for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Create three side-by-side subplots
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), subplot_kw=dict(projection='polar'))
            
            # Function to plot radar for a given metric
            def plot_radar(ax, data, title, metric_name):
                if data.empty:
                    ax.text(0.5, 0.5, f'No {metric_name} data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(title, size=14, fontweight='bold', pad=20)
                    return
                
                # Color map for different models
                colors = plt.cm.Set1(np.linspace(0, 1, len(data.index)))
                
                # Plot each model
                for i, (model_name, values) in enumerate(data.iterrows()):
                    # Fill NaN values with 0 for plotting
                    values_list = [values[cat] if not pd.isna(values[cat]) else 0 for cat in categories]
                    values_list += values_list[:1]  # Complete the circle
                    
                    ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name, color=colors[i])
                    ax.fill(angles, values_list, alpha=0.1, color=colors[i])
                
                # Customize the plot
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                ax.grid(True)
                ax.set_title(title, size=14, fontweight='bold', pad=20)
                
                # Add individual legend for this radar plot
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            
            # Extract context length for title
            context_clean = max_len.replace('max_len_', '')
            
            # Plot all three metrics
            plot_radar(ax1, accuracy_data, f'Accuracy ({context_clean} tokens)', 'accuracy')
            plot_radar(ax2, f1_data, f'F1 ({context_clean} tokens)', 'f1_score')
            plot_radar(ax3, recall_data, f'Recall ({context_clean} tokens)', 'recall_score')
            
            plt.tight_layout()
            
            # Save plot with max_len in filename
            output_file = self.output_dir / f"radar_plot_combined_{max_len}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Radar plot saved: {output_file}")
            saved_files.append(str(output_file))
        
        return saved_files

    def create_performance_overview(self, df: pd.DataFrame) -> Optional[str]:
        """Create a comprehensive overview dashboard."""
        
        # Create a figure with 2x2 layout
        plt.figure(figsize=(16, 10))
        
        # Group-level data
        group_df = df[df['task_level'] == 'group'].copy()
        
        if group_df.empty:
            print("No group-level data for overview")
            return None
        
        # Subplot 1: Parent group performance (both accuracy and F1)
        ax1 = plt.subplot(2, 2, 1)
        if 'parent_group' in group_df.columns:
            parent_avg = group_df.groupby('parent_group').agg({
                'accuracy': 'mean',
                'f1_score': 'mean'
            })
            # Parent groups are already clean from extraction, just format for display
            parent_avg.index = parent_avg.index.str.replace('_', ' ').str.title()
            
            parent_avg.plot(kind='bar', ax=ax1)
            ax1.set_ylabel('Performance')
            ax1.set_title('Performance by Parent Group')
            ax1.tick_params(axis='x', rotation=20)
            ax1.legend(['Accuracy', 'F1 Score'])
        else:
            ax1.text(0.5, 0.5, 'Parent Group\nData Not Available', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Parent Group Analysis')
        
        # Subplot 2: Model performance (both accuracy and F1)
        ax2 = plt.subplot(2, 2, 2)
        model_avg = group_df.groupby('model_name').agg({
            'accuracy': 'mean',
            'f1_score': 'mean'
        })
        # Clean up model labels
        model_avg.index = model_avg.index.str.replace('Qwen__', '', regex=False)
        model_avg.index = model_avg.index.str.replace('meta-llama__', '', regex=False)
        
        model_avg.plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Performance')
        ax2.set_title('Performance by Model')
        ax2.tick_params(axis='x', rotation=20)
        ax2.legend(['Accuracy', 'F1 Score'])
        
        # Subplot 3: Accuracy vs F1 scatter (colored by parent group)
        ax3 = plt.subplot(2, 2, 3)
        if 'parent_group' in group_df.columns and len(group_df['parent_group'].unique()) > 1:
            parent_groups = group_df['parent_group'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(parent_groups)))
            for i, parent_group in enumerate(parent_groups):
                group_data = group_df[group_df['parent_group'] == parent_group]
                # Parent groups are already clean from extraction, just format for display
                clean_label = parent_group.replace('_', ' ').title()
                ax3.scatter(group_data['accuracy'], group_data['f1_score'], 
                           c=[colors[i]], alpha=0.6, label=clean_label, s=50)
            # Embed legend inside the plot area
            ax3.legend(loc='upper left', fontsize='small', framealpha=0.8)
        else:
            ax3.scatter(group_df['accuracy'], group_df['f1_score'], alpha=0.6)
        
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Accuracy vs F1 Score by Parent Group')
        
        # Subplot 4: Accuracy vs F1 scatter (colored by model)
        ax4 = plt.subplot(2, 2, 4)
        if 'model_name' in group_df.columns and len(group_df['model_name'].unique()) > 1:
            models = group_df['model_name'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            for i, model in enumerate(models):
                model_data = group_df[group_df['model_name'] == model]
                # Clean up model label for legend
                clean_label = model.replace('Qwen__', '').replace('meta-llama__', '')
                ax4.scatter(model_data['accuracy'], model_data['f1_score'], 
                           c=[colors[i]], alpha=0.6, label=clean_label, s=50)
            # Embed legend inside the plot area
            ax4.legend(loc='upper left', fontsize='small', framealpha=0.8)
        else:
            ax4.scatter(group_df['accuracy'], group_df['f1_score'], alpha=0.6)
        
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Accuracy vs F1 Score by Model')
        
        plt.tight_layout()
        
        # Save plot (overwrite latest)
        output_file = self.output_dir / "performance_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance overview saved: {output_file}")
        return str(output_file)
    
    def create_individual_group_radar_plot(self, df: pd.DataFrame, parent_group: str) -> List[str]:
        """Create radar plots for individual tasks within a specific parent group, separated by max_len."""
        
        # Filter for individual-level results within this parent group
        individual_df = df[(df['task_level'] == 'individual') & (df['parent_group'] == parent_group)].copy()
        
        if individual_df.empty:
            print(f"No individual-level data for parent group: {parent_group}")
            return []
        
        # Clean up labels
        individual_df_clean = individual_df.copy()
        individual_df_clean['model_name_clean'] = individual_df_clean['model_name'].str.replace('Qwen__', '', regex=False).str.replace('meta-llama__', '', regex=False)
        
        # Get unique max_len values
        max_lens = sorted(individual_df_clean['max_len'].unique())
        saved_files = []
        
        for max_len in max_lens:
            # Filter data for this specific max_len
            max_len_data = individual_df_clean[individual_df_clean['max_len'] == max_len].copy()
            
            if max_len_data.empty:
                continue
            
            # Create pivot tables for all three metrics (using task_name as columns)
            accuracy_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='task_name',
                values='accuracy',
                aggfunc='mean'
            )
            
            f1_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='task_name',
                values='f1_score',
                aggfunc='mean'
            )
            
            recall_data = max_len_data.pivot_table(
                index='model_name_clean',
                columns='task_name',
                values='recall_score',
                aggfunc='mean'
            ) if 'recall_score' in max_len_data.columns else pd.DataFrame()
            
            if accuracy_data.empty and f1_data.empty and recall_data.empty:
                continue
            
            # Get the tasks (dimensions) - use whichever is available
            categories = list(accuracy_data.columns) if not accuracy_data.empty else (
                list(f1_data.columns) if not f1_data.empty else list(recall_data.columns)
            )
            N = len(categories)
            
            if N < 3:
                print(f"Need at least 3 tasks for radar plot in group {parent_group} (max_len: {max_len}), found {N}")
                continue
            
            # Create angles for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Create three side-by-side subplots
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), subplot_kw=dict(projection='polar'))
            
            # Function to plot radar for a given metric
            def plot_radar(ax, data, title, metric_name):
                if data.empty:
                    ax.text(0.5, 0.5, f'No {metric_name} data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(title, size=14, fontweight='bold', pad=20)
                    return
                
                # Color map for different models
                colors = plt.cm.Set1(np.linspace(0, 1, len(data.index)))
                
                # Plot each model
                for i, (model_name, values) in enumerate(data.iterrows()):
                    # Fill NaN values with 0 for plotting
                    values_list = [values[cat] if not pd.isna(values[cat]) else 0 for cat in categories]
                    values_list += values_list[:1]  # Complete the circle
                    
                    ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name, color=colors[i])
                    ax.fill(angles, values_list, alpha=0.1, color=colors[i])
                
                # Customize the plot
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                ax.grid(True)
                ax.set_title(title, size=14, fontweight='bold', pad=20)
                
                # Add individual legend for this radar plot
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            
            # Parent groups are already clean from extraction, just format for display
            clean_parent_group = parent_group.replace('_', ' ').title()
            
            # Extract context length for title
            context_clean = max_len.replace('max_len_', '')
            
            # Plot all three metrics
            plot_radar(ax1, accuracy_data, f'{clean_parent_group} - Accuracy ({context_clean} tokens)', 'accuracy')
            plot_radar(ax2, f1_data, f'{clean_parent_group} - F1 ({context_clean} tokens)', 'f1_score')
            plot_radar(ax3, recall_data, f'{clean_parent_group} - Recall ({context_clean} tokens)', 'recall_score')
            
            plt.tight_layout()
            
            # Save plot with parent group name and max_len
            clean_filename = clean_parent_group.replace(' ', '_').replace('-', '_').lower()
            output_file = self.output_dir / f"radar_plot_combined_{clean_filename}_{max_len}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Individual group radar plot saved: {output_file}")
            saved_files.append(str(output_file))
        
        return saved_files

    def generate_visualizations(self, df: pd.DataFrame, level: str = None) -> List[str]:
        """Generate appropriate visualizations based on the task level."""
        
        saved_files = []
        
        print("Generating visualizations...")
        
        # Check what data we have
        has_group_data = not df[df['task_level'] == 'group'].empty if 'task_level' in df.columns else False
        has_individual_data = not df[df['task_level'] == 'individual'].empty if 'task_level' in df.columns else False
        
        # Group-level visualizations (work with group data)
        if level == 'group':
            if not has_group_data:
                print("No group-level data available for visualization")
                return saved_files
            
            # Max length analysis (shows accuracy, F1, and recall across different context lengths)
            length_file = self.create_max_length_analysis(df)
            if length_file:
                saved_files.append(length_file)
            
            # Task type heatmap (shows accuracy, F1, and recall across task types)
            heatmap_file = self.create_task_type_heatmap(df)
            if heatmap_file:
                saved_files.append(heatmap_file)
            
            # Radar plots for model comparison (shows accuracy, F1, and recall for each max_len)
            radar_files = self.create_radar_plot(df)
            if radar_files:
                saved_files.extend(radar_files)
        
        # Individual-level visualizations (stratified by parent group)
        elif level == 'individual':
            if not has_individual_data:
                print("No individual-level data available for visualization")
                return saved_files
            
            # Get unique parent groups from individual data
            individual_df = df[df['task_level'] == 'individual']
            if 'parent_group' not in individual_df.columns:
                print("No parent_group column found in individual data")
                return saved_files
            
            # Check task counts for mortality and operational groups
            parent_group_counts = individual_df.groupby('parent_group')['task_name'].nunique()
            
            # Since parent groups are now clean, check for exact matches
            mortality_tasks = parent_group_counts.get('mortality', 0)
            operational_tasks = parent_group_counts.get('operational', 0)
            
            # Check if we need to combine mortality and operational into inpatient_related
            combine_inpatient = False
            if (mortality_tasks > 0 and mortality_tasks < 3) or (operational_tasks > 0 and operational_tasks < 3):
                combine_inpatient = True
                print(f"Combining mortality ({mortality_tasks} tasks) and operational ({operational_tasks} tasks) into inpatient_related group")
            
            # Create modified DataFrame for visualization
            viz_df = individual_df.copy()
            
            if combine_inpatient:
                # Combine mortality and operational groups into inpatient_related
                inpatient_mask = (
                    (viz_df['parent_group'] == 'mortality') |
                    (viz_df['parent_group'] == 'operational')
                )
                viz_df.loc[inpatient_mask, 'parent_group'] = 'inpatient_related'
            
            # Get final parent groups for visualization
            parent_groups = viz_df['parent_group'].unique()
            final_group_counts = viz_df.groupby('parent_group')['task_name'].nunique()
            
            print(f"Final parent groups for individual task visualization:")
            for pg in sorted(parent_groups):
                task_count = final_group_counts[pg]
                print(f"  - {pg}: {task_count} tasks")
            
            # Generate radar plots for each parent group (only those with >= 3 tasks)
            for parent_group in sorted(parent_groups):
                task_count = final_group_counts[parent_group]
                if task_count >= 3:
                    print(f"Generating radar plots for parent group: {parent_group} ({task_count} tasks)")
                    radar_files = self.create_individual_group_radar_plot(viz_df, parent_group)
                    if radar_files:
                        saved_files.extend(radar_files)
                else:
                    print(f"Skipping {parent_group} - only {task_count} tasks (need at least 3 for radar plot)")
        
        else:
            print(f"Invalid level: {level}. Must be 'group' or 'individual'")
        
        return saved_files

def main():
    """Main function to run the visualization."""
    
    parser = argparse.ArgumentParser(description='Visualize EHRShot evaluation results')
    parser.add_argument('--data_file', type=str, help='Path to extracted results CSV/JSON file')
    parser.add_argument('--output_dir', type=str, help='Path to save visualization files')
    parser.add_argument('--level', type=str, choices=['group', 'individual'], 
                       required=True, help='Filter by task level (required)')
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = EHRShotResultsVisualizer(output_dir=args.output_dir, level=args.level)
        
        # Auto-detect data file if not provided
        if args.data_file is None:
            # Look for standard result files in the extracted_results directory
            script_dir = Path(__file__).parent
            
            # Prioritize level-specific files based on the level argument
            if args.level == 'group':
                target_file = script_dir / "extracted_results" / "ehrshot_results_summary_group.csv"
            else:  # args.level == 'individual'
                target_file = script_dir / "extracted_results" / "ehrshot_results_summary_individual.csv"
            
            if target_file.exists():
                args.data_file = str(target_file)
                print(f"Auto-detected data file: {args.data_file}")
            else:
                # Fallback to any CSV files with timestamps (for backwards compatibility)
                csv_files = list(script_dir.glob("extracted_results/ehrshot_results_summary*.csv"))
                if csv_files:
                    args.data_file = str(max(csv_files, key=lambda x: x.stat().st_mtime))
                    print(f"Auto-detected data file: {args.data_file}")
                else:
                    print("No data file found. Please provide --data_file or run extract_results.py first.")
                    return
        
        # Load data
        print(f"Loading data from: {args.data_file}")
        df = visualizer.load_data(args.data_file)
        
        if df.empty:
            print("No data found in the file!")
            return
        
        print(f"Loaded {len(df)} rows of data")
        
        # Filter by level (required parameter)
        if 'task_level' in df.columns:
            df_filtered = df[df['task_level'] == args.level].copy()
            print(f"Filtered to {args.level} level: {len(df_filtered)} rows")
            if df_filtered.empty:
                print(f"No {args.level} level data found!")
                return
            df = df_filtered
        else:
            print("Warning: task_level column not found, using all data")
        
        # Generate visualizations
        saved_files = visualizer.generate_visualizations(df, level=args.level)
        
        print("\nVisualization completed successfully!")
        print(f"Generated {len(saved_files)} visualization files:")
        for file_path in saved_files:
            print(f"  - {file_path}")
        print(f"\nAll files saved to: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
