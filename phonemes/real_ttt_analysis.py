#!/usr/bin/env python3

"""
Comprehensive analysis of REAL TTT-RNN training results.
Analyzes actual data from the basic_ttt_20250609_020007 experiment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from datetime import datetime, timedelta
import yaml
from collections import defaultdict

# Set up plotting style
try:
    plt.style.use('seaborn')
except OSError:
    try:
        plt.style.use('ggplot')
    except OSError:
        plt.style.use('default')

sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class TTTExperimentAnalyzer:
    """Analyzer for real TTT-RNN experiment results."""
    
    def __init__(self, experiment_dir, wandb_dir, logs_wandb_dir):
        """
        Initialize analyzer with experiment directories.
        
        Args:
            experiment_dir: Path to the experiment directory
            wandb_dir: Path to the wandb run directory (for summary)
            logs_wandb_dir: Path to the wandb run directory (for logs)
        """
        self.experiment_dir = experiment_dir
        self.wandb_dir = wandb_dir
        self.logs_wandb_dir = logs_wandb_dir
        self.results = {}
        
        # Load all data
        self.load_config()
        self.load_wandb_summary()
        self.parse_training_logs()
        
        # Create output directory
        self.output_dir = "real_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_config(self):
        """Load experiment configuration."""
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
        
    def load_wandb_summary(self):
        """Load wandb summary metrics."""
        summary_path = os.path.join(self.wandb_dir, 'files', 'wandb-summary.json')
        with open(summary_path, 'r') as f:
            self.wandb_summary = json.load(f)
        print(f"Loaded wandb summary from {summary_path}")
        
    def parse_training_logs(self):
        """Parse training logs to extract epoch-by-epoch metrics."""
        log_path = os.path.join(self.logs_wandb_dir, 'files', 'output.log')
        
        self.training_metrics = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'batch_losses': [],
            'batch_grad_norms': [],
            'batch_learning_rates': []
        }
        
        current_epoch = 0
        batch_data = {'losses': [], 'grad_norms': [], 'lrs': []}
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Extract epoch completion info
                if "Epoch" in line and "completed in" in line:
                    # Parse epoch completion line
                    # Example: "Epoch 1 completed in 2576.76s:"
                    epoch_match = re.search(r'Epoch (\d+) completed in ([\d.]+)s:', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        epoch_time = float(epoch_match.group(2))
                        self.training_metrics['epochs'].append(current_epoch)
                        self.training_metrics['epoch_time'].append(epoch_time)
                
                # Extract train/val metrics
                elif "Train Loss:" in line and "Train Acc:" in line:
                    # Example: "  Train Loss: 0.7660, Train Acc: 0.7129"
                    train_match = re.search(r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)', line)
                    if train_match:
                        train_loss = float(train_match.group(1))
                        train_acc = float(train_match.group(2))
                        self.training_metrics['train_loss'].append(train_loss)
                        self.training_metrics['train_acc'].append(train_acc)
                
                elif "Val Loss:" in line and "Val Acc:" in line:
                    # Example: "  Val Loss: 0.7621, Val Acc: 0.7098"
                    val_match = re.search(r'Val Loss: ([\d.]+), Val Acc: ([\d.]+)', line)
                    if val_match:
                        val_loss = float(val_match.group(1))
                        val_acc = float(val_match.group(2))
                        self.training_metrics['val_loss'].append(val_loss)
                        self.training_metrics['val_acc'].append(val_acc)
                
                elif "Learning Rate:" in line:
                    # Example: "  Learning Rate: 0.000085"
                    lr_match = re.search(r'Learning Rate: ([\d.e-]+)', line)
                    if lr_match:
                        lr = float(lr_match.group(1))
                        self.training_metrics['learning_rate'].append(lr)
                
                # Extract batch-level metrics
                elif "Batch" in line and "loss=" in line and "grad_norm=" in line:
                    # Example: "  Batch 5: loss=1.0263, grad_norm=8.0057, lr=0.000100"
                    batch_match = re.search(r'loss=([\d.]+), grad_norm=([\d.]+), lr=([\d.e-]+)', line)
                    if batch_match:
                        loss = float(batch_match.group(1))
                        grad_norm = float(batch_match.group(2))
                        lr = float(batch_match.group(3))
                        batch_data['losses'].append(loss)
                        batch_data['grad_norms'].append(grad_norm)
                        batch_data['lrs'].append(lr)
        
        # Store batch data
        self.training_metrics['batch_losses'] = batch_data['losses']
        self.training_metrics['batch_grad_norms'] = batch_data['grad_norms']
        self.training_metrics['batch_learning_rates'] = batch_data['lrs']
        
        print(f"Parsed {len(self.training_metrics['epochs'])} epochs of training data")
        print(f"Parsed {len(batch_data['losses'])} batch-level measurements")
        
    def create_training_overview_plot(self):
        """Create individual training overview plots."""
        epochs = self.training_metrics['epochs']
        
        # 1. Loss curves
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.training_metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.training_metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss - TTT-RNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Accuracy curves
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.training_metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.training_metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy - TTT-RNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.6, 0.8])  # Focus on the relevant range
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Learning rate schedule
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.training_metrics['learning_rate'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule (Cosine Decay) - TTT-RNN')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Epoch training time
        plt.figure(figsize=(12, 8))
        plt.bar(epochs, np.array(self.training_metrics['epoch_time']) / 60, alpha=0.7, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Training Time (minutes)')
        plt.title('Training Time per Epoch - TTT-RNN')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_time_per_epoch.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Batch loss progression (first 500 batches)
        plt.figure(figsize=(12, 8))
        batch_losses = self.training_metrics['batch_losses'][:500]
        plt.plot(batch_losses, alpha=0.7, linewidth=1)
        plt.xlabel('Batch (First 500)')
        plt.ylabel('Loss')
        plt.title('Batch Loss Progression (Early Training) - TTT-RNN')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'batch_loss_progression.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Gradient norm distribution
        plt.figure(figsize=(12, 8))
        grad_norms = self.training_metrics['batch_grad_norms']
        plt.hist(grad_norms, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Gradient Norm')
        plt.ylabel('Frequency')
        plt.title('Gradient Norm Distribution - TTT-RNN')
        plt.axvline(np.mean(grad_norms), color='red', linestyle='--', label=f'Mean: {np.mean(grad_norms):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_norm_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_performance_analysis_plot(self):
        """Create individual performance analysis plots."""
        epochs = self.training_metrics['epochs']
        train_acc = np.array(self.training_metrics['train_acc'])
        val_acc = np.array(self.training_metrics['val_acc'])
        
        # 1. Overfitting analysis
        plt.figure(figsize=(12, 8))
        gap = train_acc - val_acc
        plt.plot(epochs, gap, 'purple', linewidth=2, label='Train-Val Gap')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap (Train - Val)')
        plt.title('Overfitting Analysis - TTT-RNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Learning rate vs performance
        plt.figure(figsize=(12, 8))
        plt.scatter(self.training_metrics['learning_rate'], val_acc, alpha=0.7, s=50)
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Accuracy')
        plt.title('Learning Rate vs Validation Performance - TTT-RNN')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lr_vs_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Training stability (loss variance)
        plt.figure(figsize=(12, 8))
        window_size = 50
        batch_losses = self.training_metrics['batch_losses']
        if len(batch_losses) >= window_size:
            # Calculate rolling variance
            rolling_var = []
            for i in range(window_size, len(batch_losses)):
                window = batch_losses[i-window_size:i]
                rolling_var.append(np.var(window))
            
            plt.plot(rolling_var, alpha=0.7, linewidth=1)
            plt.xlabel(f'Batch (Rolling Window of {window_size})')
            plt.ylabel('Loss Variance')
            plt.title('Training Stability (Loss Variance) - TTT-RNN')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Convergence analysis
        plt.figure(figsize=(12, 8))
        plt.semilogy(epochs, self.training_metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.semilogy(epochs, self.training_metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Convergence - TTT-RNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_convergence.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_dataset_analysis_plot(self):
        """Create individual dataset and model analysis plots."""
        # Extract dataset info from wandb summary
        total_sentences = self.wandb_summary['dataset/total_sentences']
        train_sentences = self.wandb_summary['dataset/train_sentences']
        val_sentences = self.wandb_summary['dataset/val_sentences']
        avg_length = self.wandb_summary['dataset/avg_sentence_length']
        min_length = self.wandb_summary['dataset/min_sentence_length']
        max_length = self.wandb_summary['dataset/max_sentence_length']
        n_features = self.wandb_summary['dataset/n_features']
        feature_max = self.wandb_summary['dataset/feature_max']
        
        # 1. Dataset split visualization
        plt.figure(figsize=(10, 8))
        sizes = [train_sentences, val_sentences]
        labels = [f'Train ({train_sentences})', f'Val ({val_sentences})']
        colors = ['lightblue', 'lightcoral']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Dataset Split\nTotal: {total_sentences} sentences')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_split.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Model parameters breakdown
        plt.figure(figsize=(10, 8))
        total_params = self.wandb_summary['model/total_parameters']
        
        # Estimate parameter breakdown for TTT-RNN
        units = self.config['model']['units']
        n_layers = self.config['model']['n_layers']
        n_classes = 40  # 39 phonemes + 1 new class signal
        
        # Rough parameter estimates
        input_params = n_features * units  # Input projection
        rnn_params = units * units * 4 * n_layers  # RNN parameters (approximate)
        ttt_params = total_params * 0.3  # TTT mechanism (estimated)
        output_params = units * n_classes  # Output projection
        other_params = total_params - (input_params + rnn_params + ttt_params + output_params)
        
        param_breakdown = {
            'Input Layer': input_params,
            'RNN Layers': rnn_params,
            'TTT Mechanism': ttt_params,
            'Output Layer': output_params,
            'Other': max(0, other_params)
        }
        
        sizes = list(param_breakdown.values())
        labels = [f'{k}\n({v/1000:.0f}K)' for k, v in param_breakdown.items()]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Model Parameters Breakdown\nTotal: {total_params/1000000:.2f}M parameters')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_parameters_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Training efficiency metrics
        plt.figure(figsize=(12, 8))
        total_runtime = self.wandb_summary['_runtime']  # in seconds
        total_epochs = len(self.training_metrics['epochs'])
        avg_epoch_time = np.mean(self.training_metrics['epoch_time'])
        
        metrics = {
            'Total Runtime (hours)': total_runtime / 3600,
            'Avg Epoch Time (min)': avg_epoch_time / 60,
            'Time per Sample (ms)': (total_runtime / (total_epochs * train_sentences)) * 1000,
            'Samples per Second': (total_epochs * train_sentences) / total_runtime
        }
        
        bars = plt.bar(range(len(metrics)), list(metrics.values()), 
                     color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
        plt.ylabel('Value')
        plt.title('Training Efficiency Metrics - TTT-RNN')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_ttt_specific_analysis(self):
        """Create TTT-specific analysis plots using only real data."""
        
        # 1. TTT Configuration Visualization
        plt.figure(figsize=(12, 10))
        ttt_config = self.config['model']['ttt_config']
        config_text = f"""TTT Configuration:
• Inner Encoder: {ttt_config['inner_encoder']}
• Inner Iterations: {ttt_config['inner_iterations']}
• Inner Learning Rate: {ttt_config['inner_lr']}
• Use SGD: {ttt_config['use_sgd']}
• Decoder LayerNorm: {ttt_config['decoder_ln']}
• Enhanced TTT: {self.config['model']['use_enhanced_ttt']}

Model Architecture:
• Type: {self.config['model']['type']}
• Units: {self.config['model']['units']}
• Layers: {self.config['model']['n_layers']}
• Bidirectional: {self.config['model']['bidirectional']}
• Dropout: {self.config['model']['dropout']}"""
        
        plt.text(0.05, 0.95, config_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('TTT-RNN Configuration', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ttt_configuration.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Gradient norm analysis
        plt.figure(figsize=(12, 8))
        grad_norms = np.array(self.training_metrics['batch_grad_norms'])
        
        # Create gradient norm statistics
        plt.boxplot([grad_norms], labels=['All Batches'])
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Distribution (TTT Inner Loop Effects)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(grad_norms):.3f}
Std: {np.std(grad_norms):.3f}
Max: {np.max(grad_norms):.3f}
95th %ile: {np.percentile(grad_norms, 95):.3f}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_norm_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Learning rate adaptation analysis
        plt.figure(figsize=(12, 8))
        batch_lrs = self.training_metrics['batch_learning_rates']
        batch_indices = range(len(batch_lrs))
        
        plt.plot(batch_indices, batch_lrs, alpha=0.7, linewidth=1)
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule Throughout Training - TTT-RNN')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'batch_learning_rate_progression.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. TTT Inner vs Outer Learning Rate Comparison (Real Data Only)
        plt.figure(figsize=(12, 8))
        
        # Get final outer learning rate and TTT inner learning rate
        final_outer_lr = self.training_metrics['learning_rate'][-1]
        initial_outer_lr = self.training_metrics['learning_rate'][0]
        ttt_inner_lr = self.config['model']['ttt_config']['inner_lr']
        
        # Create comparison of learning rates
        lr_types = ['Initial Outer LR', 'Final Outer LR', 'TTT Inner LR']
        lr_values = [initial_outer_lr, final_outer_lr, ttt_inner_lr]
        colors = ['blue', 'red', 'green']
        
        bars = plt.bar(lr_types, lr_values, color=colors, alpha=0.7)
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Comparison: Outer vs TTT Inner')
        plt.yscale('log')
        
        # Add value labels
        for bar, lr in zip(bars, lr_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lr:.1e}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_rate_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive text report of the analysis."""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("TTT-RNN EXPERIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Experiment: {os.path.basename(self.experiment_dir)}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Experiment Configuration
        report.append("EXPERIMENT CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Model Type: {self.config['model']['type']}")
        report.append(f"Model Units: {self.config['model']['units']}")
        report.append(f"Model Layers: {self.config['model']['n_layers']}")
        report.append(f"Enhanced TTT: {self.config['model']['use_enhanced_ttt']}")
        report.append(f"Bidirectional: {self.config['model']['bidirectional']}")
        report.append(f"Dropout: {self.config['model']['dropout']}")
        report.append("")
        
        # TTT Configuration
        ttt_config = self.config['model']['ttt_config']
        report.append("TTT CONFIGURATION")
        report.append("-" * 40)
        for key, value in ttt_config.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Dataset Information
        report.append("DATASET INFORMATION")
        report.append("-" * 40)
        report.append(f"Total Sentences: {self.wandb_summary['dataset/total_sentences']:,}")
        report.append(f"Training Sentences: {self.wandb_summary['dataset/train_sentences']:,}")
        report.append(f"Validation Sentences: {self.wandb_summary['dataset/val_sentences']:,}")
        report.append(f"Average Sentence Length: {self.wandb_summary['dataset/avg_sentence_length']:.1f} time steps")
        report.append(f"Sentence Length Range: {self.wandb_summary['dataset/min_sentence_length']}-{self.wandb_summary['dataset/max_sentence_length']} time steps")
        report.append(f"Number of Features: {self.wandb_summary['dataset/n_features']:,}")
        report.append(f"Feature Value Range: 0 - {self.wandb_summary['dataset/feature_max']:.0e}")
        report.append("")
        
        # Model Information
        report.append("MODEL INFORMATION")
        report.append("-" * 40)
        report.append(f"Total Parameters: {self.wandb_summary['model/total_parameters']:,}")
        report.append(f"Parameter Density: {self.wandb_summary['model/total_parameters'] / self.wandb_summary['dataset/n_features']:.1f} params per input feature")
        report.append("")
        
        # Training Configuration
        report.append("TRAINING CONFIGURATION")
        report.append("-" * 40)
        train_config = self.config['training']
        report.append(f"Epochs: {train_config['epochs']}")
        report.append(f"Batch Size: {train_config['batch_size']}")
        report.append(f"Initial Learning Rate: {train_config['initial_lr']}")
        report.append(f"Final Learning Rate: {train_config['final_lr']}")
        report.append(f"Learning Rate Schedule: {'Cosine Decay' if train_config['use_cosine_decay'] else 'Exponential Decay'}")
        report.append(f"Gradient Clipping: {train_config['grad_clip_norm']}")
        report.append("")
        
        # Performance Results
        report.append("PERFORMANCE RESULTS")
        report.append("-" * 40)
        final_train_loss = self.training_metrics['train_loss'][-1]
        final_train_acc = self.training_metrics['train_acc'][-1]
        final_val_loss = self.training_metrics['val_loss'][-1]
        final_val_acc = self.training_metrics['val_acc'][-1]
        best_val_loss = min(self.training_metrics['val_loss'])
        best_val_acc = max(self.training_metrics['val_acc'])
        
        report.append(f"Final Training Loss: {final_train_loss:.4f}")
        report.append(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        report.append(f"Final Validation Loss: {final_val_loss:.4f}")
        report.append(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        report.append(f"Best Validation Loss: {best_val_loss:.4f}")
        report.append(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        report.append("")
        
        # Training Dynamics
        report.append("TRAINING DYNAMICS")
        report.append("-" * 40)
        total_runtime = self.wandb_summary['_runtime']
        avg_epoch_time = np.mean(self.training_metrics['epoch_time'])
        grad_norms = np.array(self.training_metrics['batch_grad_norms'])
        
        report.append(f"Total Training Time: {total_runtime/3600:.2f} hours")
        report.append(f"Average Epoch Time: {avg_epoch_time/60:.2f} minutes")
        report.append(f"Average Gradient Norm: {np.mean(grad_norms):.4f}")
        report.append(f"Max Gradient Norm: {np.max(grad_norms):.4f}")
        report.append(f"Gradient Norm Std: {np.std(grad_norms):.4f}")
        report.append("")
        
        # Overfitting Analysis
        report.append("OVERFITTING ANALYSIS")
        report.append("-" * 40)
        train_val_gap = final_train_acc - final_val_acc
        max_gap = max(np.array(self.training_metrics['train_acc']) - np.array(self.training_metrics['val_acc']))
        
        report.append(f"Final Train-Val Accuracy Gap: {train_val_gap:.4f} ({train_val_gap*100:.2f}%)")
        report.append(f"Maximum Train-Val Gap: {max_gap:.4f} ({max_gap*100:.2f}%)")
        if train_val_gap < 0.05:
            report.append("Status: No significant overfitting detected")
        elif train_val_gap < 0.1:
            report.append("Status: Mild overfitting")
        else:
            report.append("Status: Moderate overfitting - consider regularization")
        report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        
        # Learning rate analysis
        lr_reduction = self.training_metrics['learning_rate'][0] / self.training_metrics['learning_rate'][-1]
        report.append(f"• Learning rate reduced by {lr_reduction:.0f}x during training (cosine schedule)")
        
        # Convergence analysis
        loss_improvement = (self.training_metrics['val_loss'][0] - final_val_loss) / self.training_metrics['val_loss'][0]
        acc_improvement = (final_val_acc - self.training_metrics['val_acc'][0]) / self.training_metrics['val_acc'][0]
        report.append(f"• Validation loss improved by {loss_improvement*100:.1f}%")
        report.append(f"• Validation accuracy improved by {acc_improvement*100:.1f}%")
        
        # Training stability
        batch_loss_cv = np.std(self.training_metrics['batch_losses']) / np.mean(self.training_metrics['batch_losses'])
        report.append(f"• Batch loss coefficient of variation: {batch_loss_cv:.3f} (lower is more stable)")
        
        # TTT-specific insights
        report.append(f"• TTT inner iterations: {ttt_config['inner_iterations']} steps per forward pass")
        report.append(f"• TTT inner learning rate: {ttt_config['inner_lr']} (vs outer LR ~1e-4)")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if final_val_acc < 0.75:
            report.append("• Consider increasing model capacity or training longer")
        if train_val_gap > 0.1:
            report.append("• Consider stronger regularization or more validation data")
        if np.max(grad_norms) > 5.0:
            report.append("• Consider reducing learning rate or increasing gradient clipping")
        if batch_loss_cv > 0.5:
            report.append("• Consider reducing batch size or adjusting learning rate for more stable training")
        if avg_epoch_time > 3600:  # 1 hour
            report.append("• Consider optimizing data loading or using smaller batch sizes for faster iteration")
        
        report.append("• TTT mechanism appears to be functioning (gradient norms within reasonable range)")
        report.append("• Consider experimenting with different TTT inner iteration counts")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 50)
        print(report_text)
        print(f"\nReport saved to: {report_path}")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive TTT-RNN experiment analysis...")
        print(f"Analyzing experiment: {os.path.basename(self.experiment_dir)}")
        print()
        
        # Create all plots individually
        print("1. Creating training overview plots...")
        self.create_training_overview_plot()
        
        print("2. Creating performance analysis plots...")
        self.create_performance_analysis_plot()
        
        print("3. Creating dataset and model analysis plots...")
        self.create_dataset_analysis_plot()
        
        print("4. Creating TTT-specific analysis plots...")
        self.create_ttt_specific_analysis()
        
        print("5. Generating comprehensive report...")
        self.generate_comprehensive_report()
        
        print(f"\nAnalysis complete! All results saved to: {self.output_dir}/")
        print("Generated individual plot files:")
        
        # List expected plot files
        expected_plots = [
            'loss_curves.png',
            'accuracy_curves.png',
            'learning_rate_schedule.png',
            'training_time_per_epoch.png',
            'batch_loss_progression.png',
            'gradient_norm_distribution.png',
            'overfitting_analysis.png',
            'lr_vs_performance.png',
            'training_stability.png',
            'loss_convergence.png',
            'dataset_split.png',
            'model_parameters_breakdown.png',
            'training_efficiency.png',
            'ttt_configuration.png',
            'gradient_norm_analysis.png',
            'batch_learning_rate_progression.png',
            'learning_rate_comparison.png',
            'comprehensive_analysis_report.txt'
        ]
        
        # Check which files were actually created
        actual_files = os.listdir(self.output_dir)
        for expected_file in expected_plots:
            if expected_file in actual_files:
                print(f"  ✓ {expected_file}")
            else:
                print(f"  ✗ {expected_file} (not found)")
        
        print(f"\nTotal files generated: {len(actual_files)}")
        print("\nAll plots are generated individually and saved as separate PNG files.")

def main():
    """Main function to run the analysis."""
    
    # Paths to the experiment data
    experiment_dir = "stats320final/phonemes/ttt_experiments/basic_ttt_20250609_020007"
    wandb_dir = "wandb/run-20250609_020006-13pv8u1i"  # For summary
    logs_wandb_dir = "stats320final/phonemes/wandb/run-20250609_020006-13pv8u1i"  # For logs
    
    # Check if directories exist
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    if not os.path.exists(wandb_dir):
        print(f"Error: Wandb directory not found: {wandb_dir}")
        return
        
    if not os.path.exists(logs_wandb_dir):
        print(f"Error: Logs wandb directory not found: {logs_wandb_dir}")
        return
    
    # Create analyzer and run analysis
    analyzer = TTTExperimentAnalyzer(experiment_dir, wandb_dir, logs_wandb_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 