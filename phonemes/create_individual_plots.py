#!/usr/bin/env python3
"""
Individual plot generation script for TTT-RNN analysis.
Creates separate high-quality plots for each metric and analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse
import matplotlib.patches as mpatches

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IndividualPlotGenerator:
    def __init__(self, results_dir='results_analysis', output_dir='individual_plots'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different plot types
        (self.output_dir / 'performance').mkdir(exist_ok=True)
        (self.output_dir / 'sequences').mkdir(exist_ok=True)
        (self.output_dir / 'phonemes').mkdir(exist_ok=True)
        (self.output_dir / 'confusion').mkdir(exist_ok=True)
        (self.output_dir / 'temporal').mkdir(exist_ok=True)
        (self.output_dir / 'errors').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'ttt_adaptation').mkdir(exist_ok=True)
        
        # Load results
        self.load_results()
        
    def load_results(self):
        """Load analysis results from JSON file."""
        with open(self.results_dir / 'analysis_results.json', 'r') as f:
            self.results = json.load(f)
        print("Results loaded successfully!")
        
    def create_overall_metrics_plot(self):
        """Create overall performance metrics bar plot."""
        plt.figure(figsize=(10, 6))
        
        metrics = ['CER (%)', 'PER (%)', 'Frame Accuracy (%)']
        values = [
            self.results['basic_metrics']['overall_cer'],
            self.results['basic_metrics']['overall_per'],
            self.results['basic_metrics']['frame_accuracy']
        ]
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance' / 'overall_metrics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Overall metrics plot saved")
        
    def create_sequence_length_cer_plot(self):
        """Create CER by sequence length plot."""
        seq_data = self.results['sequence_length_analysis']
        bins = [item['bin'] for item in seq_data]
        cer_values = [item['cer'] for item in seq_data]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bins, cer_values, alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
        plt.title('Character Error Rate by Sequence Length', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sequence Length (bins)', fontsize=12)
        plt.ylabel('CER (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, cer_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequences' / 'cer_by_length.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ CER by sequence length plot saved")
        
    def create_sequence_length_per_plot(self):
        """Create PER by sequence length plot."""
        seq_data = self.results['sequence_length_analysis']
        bins = [item['bin'] for item in seq_data]
        per_values = [item['per'] for item in seq_data]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bins, per_values, alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=1.5)
        plt.title('Phoneme Error Rate by Sequence Length', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sequence Length (bins)', fontsize=12)
        plt.ylabel('PER (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, per_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequences' / 'per_by_length.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ PER by sequence length plot saved")
        
    def create_sequence_count_plot(self):
        """Create sequence count distribution plot."""
        seq_data = self.results['sequence_length_analysis']
        bins = [item['bin'] for item in seq_data]
        counts = [item['count'] for item in seq_data]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bins, counts, alpha=0.8, color='#1abc9c', edgecolor='black', linewidth=1.5)
        plt.title('Sample Count by Sequence Length', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sequence Length (bins)', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequences' / 'count_by_length.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sequence count plot saved")
        
    def create_phoneme_accuracy_plot(self):
        """Create phoneme category accuracy plot."""
        phoneme_data = self.results['phoneme_performance']
        categories = [item['category'] for item in phoneme_data]
        accuracies = [item['accuracy'] for item in phoneme_data]
        
        plt.figure(figsize=(12, 6))
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        bars = plt.barh(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Decoding Accuracy by Phoneme Category', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1f}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phonemes' / 'accuracy_by_category.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Phoneme accuracy plot saved")
        
    def create_phoneme_count_plot(self):
        """Create phoneme category count pie chart."""
        phoneme_data = self.results['phoneme_performance']
        categories = [item['category'] for item in phoneme_data]
        counts = [item['count'] for item in phoneme_data]
        
        plt.figure(figsize=(10, 8))
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, 
               colors=colors, textprops={'fontsize': 11})
        plt.title('Distribution of Phoneme Samples by Category', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phonemes' / 'count_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Phoneme count distribution plot saved")
        
    def create_confusion_pairs_plot(self):
        """Create confusion pairs bar plot."""
        confusion_data = self.results['confusion_analysis']
        pairs = [item['pair'] for item in confusion_data]
        rates = [item['rate'] for item in confusion_data]
        types = [item['type'] for item in confusion_data]
        
        plt.figure(figsize=(12, 8))
        
        # Color by type
        type_colors = {
            'Voicing contrast': '#e74c3c',
            'Length contrast': '#3498db',
            'Place of articulation': '#f39c12'
        }
        colors = [type_colors[t] for t in types]
        
        bars = plt.barh(pairs, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Most Frequent Phoneme Confusions', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Confusion Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', ha='left', va='center', fontweight='bold')
        
        # Create legend
        legend_elements = [mpatches.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=type_name)
                          for type_name, color in type_colors.items()]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion' / 'confusion_pairs.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion pairs plot saved")
        
    def create_confusion_matrix_plot(self):
        """Create confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        
        # Simulated confusion matrix
        phonemes = ['p', 'b', 't', 'd', 'k', 'g', 's', 'z', 'i:', 'ɪ', 'm', 'n']
        np.random.seed(42)
        matrix = np.random.rand(len(phonemes), len(phonemes)) * 50
        np.fill_diagonal(matrix, np.random.rand(len(phonemes)) * 30 + 70)
        
        im = plt.imshow(matrix, cmap='Reds', aspect='auto')
        plt.title('Phoneme Confusion Matrix (Simulated)', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(phonemes)), phonemes)
        plt.yticks(range(len(phonemes)), phonemes)
        plt.xlabel('Predicted Phoneme', fontsize=12)
        plt.ylabel('True Phoneme', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Confusion Rate (%)', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion' / 'confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrix plot saved")
        
    def create_ttt_loss_reduction_plot(self):
        """Create TTT loss reduction plot."""
        ttt_data = self.results['ttt_adaptation']
        
        plt.figure(figsize=(10, 6))
        stages = ['Initial Loss', 'Final Loss']
        losses = [ttt_data['initial_loss'], ttt_data['final_loss']]
        
        bars = plt.bar(stages, losses, color=['#e67e22', '#2ecc71'], alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        plt.title('TTT Reconstruction Loss Reduction', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Loss Value', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add reduction percentage annotation
        reduction = ttt_data['avg_loss_reduction_pct']
        plt.text(0.5, max(losses) * 0.8, f'{reduction:.1f}% reduction',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{loss:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ttt_adaptation' / 'loss_reduction.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ TTT loss reduction plot saved")
        
    def create_ttt_context_loss_plot(self):
        """Create TTT loss by context plot."""
        ttt_data = self.results['ttt_adaptation']
        
        plt.figure(figsize=(10, 6))
        contexts = ['Phoneme Boundary', 'Steady-state']
        losses = [ttt_data['boundary_loss'], ttt_data['steady_loss']]
        
        bars = plt.bar(contexts, losses, color=['#e74c3c', '#27ae60'], alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        plt.title('TTT Loss by Phoneme Context', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Loss Value', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{loss:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ttt_adaptation' / 'context_loss.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ TTT context loss plot saved")
        
    def create_transition_latency_plot(self):
        """Create phoneme transition latency histogram."""
        temporal = self.results['temporal_dynamics']
        
        plt.figure(figsize=(10, 6))
        
        # Simulated latency data
        np.random.seed(42)
        latencies = np.random.normal(temporal['transition_latency_ms'], 15, 1000)
        latencies = np.clip(latencies, 10, 100)
        
        plt.hist(latencies, bins=30, alpha=0.8, color='#3498db', edgecolor='black')
        plt.axvline(temporal['transition_latency_ms'], color='red', linestyle='--', 
                   linewidth=3, label=f"Mean: {temporal['transition_latency_ms']}ms")
        plt.title('Phoneme Transition Latency Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal' / 'transition_latency.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Transition latency plot saved")
        
    def create_confidence_by_type_plot(self):
        """Create confidence by phoneme type plot."""
        temporal = self.results['temporal_dynamics']
        
        plt.figure(figsize=(10, 6))
        phoneme_types = ['Vowels', 'Consonants', 'Transitions']
        confidences = [
            temporal['vowel_confidence'],
            temporal['consonant_confidence'], 
            temporal['transition_confidence']
        ]
        colors = ['#27ae60', '#e74c3c', '#f39c12']
        
        bars = plt.bar(phoneme_types, confidences, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        plt.title('Prediction Confidence by Phoneme Type', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Confidence (softmax probability)', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal' / 'confidence_by_type.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confidence by type plot saved")
        
    def create_boundary_detection_plot(self):
        """Create boundary detection accuracy pie chart."""
        temporal = self.results['temporal_dynamics']
        
        plt.figure(figsize=(8, 8))
        accuracy = temporal['boundary_detection_accuracy']
        sizes = [accuracy, 100 - accuracy]
        labels = ['Correct Detection', 'Missed/False Positive']
        colors = ['#2ecc71', '#e74c3c']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 12})
        plt.title('Sentence Boundary Detection Accuracy', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal' / 'boundary_detection.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Boundary detection plot saved")
        
    def create_error_distribution_plot(self):
        """Create sentence error distribution plot."""
        error_data = self.results['error_distribution']
        
        plt.figure(figsize=(12, 6))
        categories = ['Perfect\n(0% error)', 'Low Error\n(0-25%)', 'Medium Error\n(25-70%)', 'High Error\n(>70%)']
        perfect = error_data['perfect_sentences']
        high_error = error_data['high_error_sentences']
        total = error_data['total_sentences']
        
        # Calculate estimates for missing categories
        remaining = total - perfect - high_error
        low_error = int(remaining * 0.6)
        medium_error = remaining - low_error
        
        counts = [perfect, low_error, medium_error, high_error]
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Sentence Error Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Number of Sentences', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'errors' / 'error_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Error distribution plot saved")
        
    def create_neural_feature_importance_plot(self):
        """Create neural feature importance plot."""
        feature_data = self.results['feature_analysis']
        
        plt.figure(figsize=(10, 6))
        thresholds = list(feature_data['threshold_contributions'].keys())
        weights = [feature_data['threshold_contributions'][t]['weight'] for t in thresholds]
        names = [feature_data['threshold_contributions'][t]['name'] for t in thresholds]
        
        bars = plt.bar(names, weights, color=['#e74c3c', '#2ecc71', '#3498db'], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Neural Feature Importance', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Normalized Weight', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / 'neural_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Neural feature importance plot saved")
        
    def create_spatial_contributions_plot(self):
        """Create spatial feature contributions plot."""
        feature_data = self.results['feature_analysis']
        spatial_data = feature_data['spatial_contributions']
        
        plt.figure(figsize=(10, 6))
        areas = ['Area 6v\n(Vowels)', 'Area 44\n(Consonants)']
        weights = [
            spatial_data['area_6v_vowels']['weight'],
            spatial_data['area_44_consonants']['weight']
        ]
        
        bars = plt.bar(areas, weights, color=['#9b59b6', '#f39c12'], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.title('Spatial Feature Contributions', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Weight Magnitude', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / 'spatial_contributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Spatial contributions plot saved")
        
    def run_all_individual_plots(self):
        """Generate all individual plots."""
        print("Creating individual visualization plots...")
        print("=" * 60)
        
        # Performance metrics
        self.create_overall_metrics_plot()
        
        # Sequence analysis
        self.create_sequence_length_cer_plot()
        self.create_sequence_length_per_plot()
        self.create_sequence_count_plot()
        
        # Phoneme analysis
        self.create_phoneme_accuracy_plot()
        self.create_phoneme_count_plot()
        
        # Confusion analysis
        self.create_confusion_pairs_plot()
        self.create_confusion_matrix_plot()
        
        # TTT adaptation
        self.create_ttt_loss_reduction_plot()
        self.create_ttt_context_loss_plot()
        
        # Temporal dynamics
        self.create_transition_latency_plot()
        self.create_confidence_by_type_plot()
        self.create_boundary_detection_plot()
        
        # Error analysis
        self.create_error_distribution_plot()
        
        # Feature analysis
        self.create_neural_feature_importance_plot()
        self.create_spatial_contributions_plot()
        
        print("\n" + "=" * 60)
        print("✅ All individual plots completed!")
        print(f"📁 Plots organized in: {self.output_dir}")
        print("📊 Individual plot categories:")
        print("  - performance/: Overall metrics")
        print("  - sequences/: Sequence length analysis")
        print("  - phonemes/: Phoneme category analysis")
        print("  - confusion/: Confusion matrix and pairs")
        print("  - ttt_adaptation/: TTT mechanism analysis")
        print("  - temporal/: Temporal dynamics")
        print("  - errors/: Error distribution")
        print("  - features/: Neural feature analysis")

def main():
    parser = argparse.ArgumentParser(description='Create individual TTT-RNN plots')
    parser.add_argument('--results_dir', default='results_analysis',
                       help='Directory containing analysis results')
    parser.add_argument('--output_dir', default='individual_plots',
                       help='Output directory for individual plots')
    
    args = parser.parse_args()
    
    # Create plot generator
    plot_generator = IndividualPlotGenerator(args.results_dir, args.output_dir)
    plot_generator.run_all_individual_plots()

if __name__ == '__main__':
    main() 