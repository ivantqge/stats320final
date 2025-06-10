#!/usr/bin/env python3
"""
Real TTT-RNN Model Analysis Script
Loads the actual trained model and evaluates performance on real data.
"""

import os
import sys
import json
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

# Add the speechBCI directory to path for imports
sys.path.append('speechBCI')
sys.path.append('speechBCI/NeuralDecoder')

# Import the TTT model and data loader
from speechBCI.NeuralDecoder.neuralDecoder.ttt_models import TTT_RNN
from neural_data_loader import NeuralDataLoader

class RealTTTAnalyzer:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        
        # Load configuration
        with open(self.experiment_dir / 'config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded experiment configuration from {experiment_dir}")
        print(f"Model type: {self.config['model']['type']}")
        print(f"Units: {self.config['model']['units']}")
        print(f"TTT Config: {self.config['model']['ttt_config']}")
        
    def load_model(self):
        """Load the trained TTT-RNN model from checkpoint."""
        print("Loading trained model...")
        
        # Get model parameters from config
        model_config = self.config['model']
        
        # Create model instance
        self.model = TTT_RNN(
            units=model_config['units'],
            weightReg=model_config['weight_reg'],
            actReg=model_config['act_reg'],
            subsampleFactor=model_config['subsample_factor'],
            nClasses=40,  # From wandb config
            bidirectional=model_config.get('bidirectional', False),
            dropout=model_config.get('dropout', 0.0),
            nLayers=model_config.get('n_layers', 2),
            ttt_config=model_config.get('ttt_config', {}),
            use_enhanced_ttt=model_config.get('use_enhanced_ttt', False)
        )
        
        # Load weights from checkpoint
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
        
        if latest_checkpoint:
            print(f"Loading weights from: {latest_checkpoint}")
            
            # We need to build the model first with a sample input
            sample_input = tf.random.normal((1, 100, 20))  # batch, time, features
            _ = self.model(sample_input)  # Build the model
            
            # Load the weights
            self.model.load_weights(latest_checkpoint)
            print("✓ Model weights loaded successfully!")
        else:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
            
    def load_data(self):
        """Load the validation dataset."""
        print("Loading validation data...")
        
        data_config = self.config['data']
        
        # Initialize data loader
        self.data_loader = NeuralDataLoader(
            data_dir=data_config['data_dir'],
            use_spikepow=data_config.get('use_spikepow', True),
            use_tx1=data_config.get('use_tx1', True),
            use_tx2=data_config.get('use_tx2', True),
            use_tx3=data_config.get('use_tx3', True),
            use_tx4=data_config.get('use_tx4', True),
            subsample_factor=data_config.get('subsample_factor', 4),
            min_sentence_length=data_config.get('min_sentence_length', 10),
            max_sentence_length=data_config.get('max_sentence_length', 200)
        )
        
        # Load and split data
        features_list, targets_list = self.data_loader.load_all_data(
            max_files=data_config.get('max_files', None)
        )
        
        if not features_list:
            raise ValueError("No data loaded! Check data directory path.")
        
        # Split into train/validation
        train_split = data_config.get('train_split', 0.8)
        split_idx = int(len(features_list) * train_split)
        
        # Use validation set for evaluation
        self.val_features = features_list[split_idx:]
        self.val_targets = targets_list[split_idx:]
        
        print(f"✓ Loaded {len(self.val_features)} validation samples")
        
    def evaluate_model(self):
        """Evaluate the model on validation data."""
        print("Evaluating model performance...")
        
        all_predictions = []
        all_targets = []
        all_cer_scores = []
        all_per_scores = []
        sentence_lengths = []
        frame_accuracies = []
        
        # Process each validation sample
        for i, (features, targets) in enumerate(tqdm(zip(self.val_features, self.val_targets), 
                                                    total=len(self.val_features), 
                                                    desc="Evaluating")):
            
            # Add batch dimension
            features_batch = tf.expand_dims(features, 0)
            
            # Get model predictions
            try:
                predictions = self.model(features_batch, training=False)
                predictions = tf.nn.softmax(predictions, axis=-1)
                pred_classes = tf.argmax(predictions, axis=-1).numpy()[0]
                
                # Get target classes
                target_classes = np.argmax(targets['class_labels_onehot'], axis=-1)
                
                # Store results
                all_predictions.append(pred_classes)
                all_targets.append(target_classes)
                sentence_lengths.append(len(pred_classes))
                
                # Compute frame accuracy
                frame_acc = np.mean(pred_classes == target_classes) * 100
                frame_accuracies.append(frame_acc)
                
                # Compute CER and PER (simplified)
                cer = self.compute_error_rate(pred_classes, target_classes)
                per = self.compute_phoneme_error_rate(pred_classes, target_classes)
                
                all_cer_scores.append(cer)
                all_per_scores.append(per)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
            # Limit evaluation for speed (remove this for full evaluation)
            if i >= 200:  # Evaluate first 200 samples for now
                break
        
        # Compute overall metrics
        self.results = {
            'basic_metrics': {
                'overall_cer': float(np.mean(all_cer_scores)),
                'overall_per': float(np.mean(all_per_scores)),
                'frame_accuracy': float(np.mean(frame_accuracies)),
                'total_samples': len(all_cer_scores)
            },
            'sentence_lengths': sentence_lengths,
            'frame_accuracies': frame_accuracies,
            'cer_scores': all_cer_scores,
            'per_scores': all_per_scores
        }
        
        print(f"\n📊 REAL MODEL PERFORMANCE:")
        print(f"Frame Accuracy: {self.results['basic_metrics']['frame_accuracy']:.1f}%")
        print(f"Character Error Rate: {self.results['basic_metrics']['overall_cer']:.1f}%")
        print(f"Phoneme Error Rate: {self.results['basic_metrics']['overall_per']:.1f}%")
        print(f"Evaluated on {self.results['basic_metrics']['total_samples']} samples")
        
    def compute_error_rate(self, predictions, targets):
        """Compute character error rate."""
        # Simple edit distance approximation
        errors = np.sum(predictions != targets)
        total = len(targets)
        return (errors / total) * 100 if total > 0 else 0
        
    def compute_phoneme_error_rate(self, predictions, targets):
        """Compute phoneme error rate."""
        # For now, same as CER - can be enhanced with phoneme groupings
        return self.compute_error_rate(predictions, targets)
        
    def analyze_by_sequence_length(self):
        """Analyze performance by sequence length."""
        lengths = np.array(self.results['sentence_lengths'])
        cer_scores = np.array(self.results['cer_scores'])
        per_scores = np.array(self.results['per_scores'])
        
        # Create length bins
        bins = [(10, 50), (51, 100), (101, 150), (151, 200)]
        sequence_analysis = []
        
        for min_len, max_len in bins:
            mask = (lengths >= min_len) & (lengths <= max_len)
            if np.any(mask):
                bin_data = {
                    'bin': f"{min_len}-{max_len}",
                    'count': int(np.sum(mask)),
                    'cer': float(np.mean(cer_scores[mask])),
                    'per': float(np.mean(per_scores[mask]))
                }
                sequence_analysis.append(bin_data)
        
        self.results['sequence_length_analysis'] = sequence_analysis
        
    def save_results(self, output_dir='real_analysis_results'):
        """Save analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Analyze by sequence length
        self.analyze_by_sequence_length()
        
        # Save detailed results
        with open(output_path / 'real_analysis_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Create summary report
        with open(output_path / 'performance_summary.txt', 'w') as f:
            f.write("TTT-RNN Real Model Performance Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Frame Accuracy: {self.results['basic_metrics']['frame_accuracy']:.2f}%\n")
            f.write(f"Character Error Rate: {self.results['basic_metrics']['overall_cer']:.2f}%\n")
            f.write(f"Phoneme Error Rate: {self.results['basic_metrics']['overall_per']:.2f}%\n")
            f.write(f"Total Samples Evaluated: {self.results['basic_metrics']['total_samples']}\n\n")
            
            f.write("Performance by Sequence Length:\n")
            for bin_data in self.results['sequence_length_analysis']:
                f.write(f"  {bin_data['bin']}: CER={bin_data['cer']:.1f}%, "
                       f"PER={bin_data['per']:.1f}%, Count={bin_data['count']}\n")
        
        print(f"✓ Results saved to {output_path}")
        
    def create_performance_plots(self, output_dir='real_analysis_results'):
        """Create performance visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Performance by sequence length
        if 'sequence_length_analysis' in self.results:
            seq_data = self.results['sequence_length_analysis']
            bins = [item['bin'] for item in seq_data]
            cer_values = [item['cer'] for item in seq_data]
            per_values = [item['per'] for item in seq_data]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # CER by length
            ax1.bar(bins, cer_values, color='#3498db', alpha=0.7)
            ax1.set_title('CER by Sequence Length (Real Model)')
            ax1.set_ylabel('CER (%)')
            ax1.set_xlabel('Sequence Length Bins')
            for i, v in enumerate(cer_values):
                ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
            
            # PER by length
            ax2.bar(bins, per_values, color='#e74c3c', alpha=0.7)
            ax2.set_title('PER by Sequence Length (Real Model)')
            ax2.set_ylabel('PER (%)')
            ax2.set_xlabel('Sequence Length Bins')
            for i, v in enumerate(per_values):
                ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_by_length.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"✓ Performance plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze real TTT-RNN model performance')
    parser.add_argument('--experiment_dir', 
                       default='ttt_experiments/basic_ttt_20250609_020007',
                       help='Path to experiment directory')
    parser.add_argument('--output_dir', 
                       default='real_analysis_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔬 TTT-RNN Real Model Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RealTTTAnalyzer(args.experiment_dir)
    
    try:
        # Load model and data
        analyzer.load_model()
        analyzer.load_data()
        
        # Evaluate performance
        analyzer.evaluate_model()
        
        # Save results and create plots
        analyzer.save_results(args.output_dir)
        analyzer.create_performance_plots(args.output_dir)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 