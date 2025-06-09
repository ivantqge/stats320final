#!/usr/bin/env python3

"""
Improved training script for TTT-RNN on neural data for sentence decoding.
Based on the original neural sequence decoder but with TTT-RNN architecture.
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import yaml
from datetime import datetime
import scipy.io
import wandb

# Enable eager execution to avoid graph mode issues
tf.config.run_functions_eagerly(True)

# Add paths for speechBCI modules
speechbci_path = os.path.join(os.path.dirname(__file__), 'speechBCI/NeuralDecoder')
sys.path.append(speechbci_path)

from neural_data_loader import NeuralDataLoader, CMU_PHONEMES
from neuralDecoder.ttt_models import TTT_RNN
from neuralDecoder.models import GRU

# Ensure GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

class NeuralTrainer:
    """
    Trainer for TTT-RNN on neural data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize wandb
        self.setup_wandb()
        
        self.setup_output_dir()
        self.setup_data_loader()
        self.setup_model()
        self.setup_training()
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb_config = self.config.get('wandb', {})
        
        # Check if wandb is enabled
        if not wandb_config.get('enabled', True):
            print("Wandb logging disabled")
            return
            
        # Create a unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "enhanced_ttt" if self.config['model']['use_enhanced_ttt'] else "basic_ttt"
        run_name = f"{model_name}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=wandb_config.get('project', 'neural-ttt-decoding'),
            entity=wandb_config.get('entity', None),
            name=run_name,
            config=self.config,
            mode=wandb_config.get('mode', 'online'),
            save_code=wandb_config.get('save_code', True),
            tags=[
                model_name,
                f"layers_{self.config['model']['n_layers']}",
                f"units_{self.config['model']['units']}",
                "neural_decoding",
                "phoneme_classification"
            ],
            notes=f"TTT-RNN training on neural data for phoneme decoding. Model: {model_name}"
        )
        
        # Log additional metadata
        wandb.config.update({
            "total_phonemes": len(CMU_PHONEMES),
            "output_classes": len(CMU_PHONEMES) + 1,  # +1 for new class signal
            "timestamp": timestamp
        })
        
        print(f"Initialized wandb run: {run_name}")
        
    def log_to_wandb(self, metrics):
        """Safely log metrics to wandb."""
        if wandb.run is not None:
            wandb.log(metrics)
        
    def log_model_info(self):
        """Log model architecture information, handling unbuilt models."""
        try:
            if hasattr(self.model, 'trainable_variables') and self.model.trainable_variables:
                total_params = sum([np.prod(v.shape) for v in self.model.trainable_variables])
                self.log_to_wandb({"model/total_parameters": total_params})
                print(f"Model has {total_params:,} trainable parameters")
            else:
                print("Model parameters not yet available (model not built)")
                self.log_to_wandb({"model/total_parameters": 0})
        except Exception as e:
            print(f"Could not count model parameters: {e}")
            self.log_to_wandb({"model/total_parameters": 0})
        
    def setup_output_dir(self):
        """Setup output directory for saving results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "enhanced_ttt" if self.config['model']['use_enhanced_ttt'] else "basic_ttt"
        
        self.output_dir = os.path.join(
            self.config['training']['output_dir'],
            f"{model_name}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Log config file to wandb
        wandb.save(config_path)
            
        print(f"Output directory: {self.output_dir}")
        
        # Log output directory to wandb
        wandb.config.update({"output_dir": self.output_dir})
        
    def setup_data_loader(self):
        """Setup the neural data loader."""
        data_config = self.config['data']
        
        self.data_loader = NeuralDataLoader(
            data_dir=data_config['data_dir'],
            use_spikepow=data_config['use_spikepow'],
            use_tx1=data_config['use_tx1'],
            use_tx2=data_config['use_tx2'],
            use_tx3=data_config['use_tx3'],
            use_tx4=data_config['use_tx4'],
            subsample_factor=data_config['subsample_factor'],
            min_sentence_length=data_config['min_sentence_length'],
            max_sentence_length=data_config['max_sentence_length']
        )
        
        print(f"Data loader setup with {self.data_loader.n_features} features")
        
    def setup_model(self):
        """Setup the TTT-RNN model."""
        model_config = self.config['model']
        
        if model_config['type'] == 'TTT_RNN':
            self.model = TTT_RNN(
                units=model_config['units'],
                weightReg=model_config['weight_reg'],
                actReg=model_config['act_reg'],
                subsampleFactor=model_config['subsample_factor'],
                nClasses=len(CMU_PHONEMES) + 1,  # +1 for new class signal
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout'],
                nLayers=model_config['n_layers'],
                ttt_config=model_config['ttt_config'],
                use_enhanced_ttt=model_config['use_enhanced_ttt']
            )
        elif model_config['type'] == 'GRU':
            self.model = GRU(
                units=model_config['units'],
                weightReg=model_config['weight_reg'],
                actReg=model_config['act_reg'],
                subsampleFactor=model_config['subsample_factor'],
                nClasses=len(CMU_PHONEMES) + 1,  # +1 for new class signal
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout'],
                nLayers=model_config['n_layers']
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
            
        # Build model with sample input
        # sample_input = tf.keras.Input(shape=(None, self.data_loader.n_features))
        # self.model(sample_input)
        
        # Let the model build itself during first training step
        print(f"Model setup: {model_config['type']} created (will build on first forward pass)")
        
    def setup_training(self):
        """Setup training components."""
        train_config = self.config['training']
        
        # Simplified learning rate schedule (removing warmup complexity for now)
        if train_config['use_cosine_decay']:
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=train_config['initial_lr'],
                decay_steps=train_config['decay_steps'],
                alpha=train_config['final_lr'] / train_config['initial_lr']
            )
        else:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=train_config['initial_lr'],
                decay_steps=train_config['decay_steps'],
                decay_rate=train_config['decay_rate']
            )
        
        # Optimizer with weight decay if specified
        if train_config.get('weight_decay', 0) > 0:
            # Try experimental AdamW first, fall back to Adam if not available
            try:
                self.optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=self.lr_schedule,
                    weight_decay=train_config['weight_decay'],
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
            except AttributeError:
                # If AdamW not available, use Adam (weight decay will be handled separately if needed)
                print("AdamW not available, using Adam optimizer instead")
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.lr_schedule,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
        
        # Loss function - match the original implementation exactly
        # Use CategoricalCrossentropy like the original, not SparseCategoricalCrossentropy
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')  # Changed to Categorical
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')  # Changed to Categorical
        
        # Setup checkpoints
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
            step=tf.Variable(0, dtype=tf.int64)
        )
        
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=os.path.join(self.output_dir, 'checkpoints'),
            max_to_keep=3
        )
        
    def load_data(self):
        """Load and prepare the data."""
        print("Loading neural data...")
        
        # Load data from specified number of files
        max_files = self.config['data'].get('max_files', None)
        features_list, targets_list = self.data_loader.load_all_data(max_files=max_files)
        
        if not features_list:
            raise ValueError("No data loaded!")
            
        # Calculate and log dataset statistics
        avg_sentence_length = np.mean([f.shape[0] for f in features_list])
        min_sentence_length = np.min([f.shape[0] for f in features_list])
        max_sentence_length = np.max([f.shape[0] for f in features_list])
        feature_min = np.min([f.min() for f in features_list])
        feature_max = np.max([f.max() for f in features_list])
        
        print(f"Loaded {len(features_list)} sentences")
        print(f"Average sentence length: {avg_sentence_length:.1f} time steps")
        print(f"Feature range: [{feature_min:.3f}, {feature_max:.3f}]")
        
        # Log dataset statistics to wandb
        self.log_to_wandb({
            "dataset/total_sentences": len(features_list),
            "dataset/avg_sentence_length": avg_sentence_length,
            "dataset/min_sentence_length": min_sentence_length,
            "dataset/max_sentence_length": max_sentence_length,
            "dataset/feature_min": feature_min,
            "dataset/feature_max": feature_max,
            "dataset/n_features": self.data_loader.n_features
        })
        
        # Split into train/val
        split_idx = int(len(features_list) * self.config['data']['train_split'])
        
        train_features = features_list[:split_idx]
        train_targets = targets_list[:split_idx]
        val_features = features_list[split_idx:]
        val_targets = targets_list[split_idx:]
        
        print(f"Train: {len(train_features)} sentences, Val: {len(val_features)} sentences")
        
        # Log train/val split info
        self.log_to_wandb({
            "dataset/train_sentences": len(train_features),
            "dataset/val_sentences": len(val_features),
            "dataset/train_split_ratio": self.config['data']['train_split']
        })
        
        # Create TensorFlow datasets
        batch_size = self.config['training']['batch_size']
        
        self.train_dataset = self.data_loader.create_tensorflow_dataset(
            train_features, train_targets, 
            batch_size=batch_size, 
            shuffle=True,
            buffer_size=min(1000, len(train_features))
        )
        
        self.val_dataset = self.data_loader.create_tensorflow_dataset(
            val_features, val_targets,
            batch_size=batch_size,
            shuffle=False
        )
        
        print("Data loading complete!")
        
    def normalize_features(self, features):
        """Normalize features to improve training stability."""
        # Neural data has huge dynamic range (0 to 11M), need robust normalization
        
        # First, clip extreme outliers (above 99th percentile per feature)
        percentile_99 = tf.reduce_max(features, axis=(0, 1), keepdims=True) * 0.99
        features = tf.clip_by_value(features, 0.0, percentile_99)
        
        # Apply square root transformation to reduce dynamic range (better than log for neural data)
        sqrt_features = tf.math.sqrt(features + 1e-6)  # Add small constant for numerical stability
        
        # Robust normalization using mean and std but with clipping
        mean = tf.reduce_mean(sqrt_features, axis=(0, 1), keepdims=True)
        std = tf.math.reduce_std(sqrt_features, axis=(0, 1), keepdims=True)
        
        # Normalize with protection against division by zero
        normalized = (sqrt_features - mean) / (std + 1e-6)
        
        # Final clipping to prevent extreme values that could cause NaN
        normalized = tf.clip_by_value(normalized, -10.0, 10.0)
        
        return normalized
        
    @tf.function
    def train_step(self, features, targets):
        """Single training step."""
        # Unpack targets
        class_labels_onehot, ce_mask, new_class_signal = targets
        
        # Normalize features
        features = self.normalize_features(features)
        
        # Check for NaN in features after normalization
        if tf.reduce_any(tf.math.is_nan(features)):
            tf.print("NaN detected in normalized features!")
            
        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            
            # Check for NaN in predictions
            if tf.reduce_any(tf.math.is_nan(predictions)):
                tf.print("NaN detected in predictions!")
            
            # Compute loss exactly like the original implementation
            # Separate phoneme prediction (first nClasses outputs) from new class signal (last output)
            n_classes = len(CMU_PHONEMES)
            
            # Create mask: tile ce_mask to match prediction dimensions
            mask = tf.tile(ce_mask[:, :, tf.newaxis], [1, 1, n_classes])
            
            # Phoneme classification loss (categorical crossentropy with mask)
            phoneme_pred = predictions[:, :, 0:n_classes] * mask
            phoneme_targets = class_labels_onehot[:, :, 0:n_classes]
            
            pred_loss = self.loss_fn(phoneme_targets, phoneme_pred)
            
            # New class signal loss (MSE between sigmoid(prediction) and target)
            new_class_pred = tf.math.sigmoid(predictions[:, :, -1])
            new_class_error = tf.reduce_mean(tf.math.square(new_class_pred - new_class_signal))
            
            # Total prediction loss
            pred_loss += new_class_error
            
            # Check for NaN in loss
            if tf.math.is_nan(pred_loss):
                tf.print("NaN detected in prediction loss!")
                tf.print("Features min/max:", tf.reduce_min(features), tf.reduce_max(features))
                tf.print("Predictions min/max:", tf.reduce_min(predictions), tf.reduce_max(predictions))
            
            # Add regularization losses
            regularization_loss = tf.constant(0.0)
            if self.model.losses:
                reg_loss = tf.add_n(self.model.losses)
                if tf.math.is_nan(reg_loss):
                    tf.print("NaN detected in regularization loss!")
                regularization_loss = reg_loss
                
            total_loss = pred_loss + regularization_loss
        
        # Compute gradients and apply
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Check for NaN in gradients
        for i, grad in enumerate(gradients):
            if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                tf.print(f"NaN detected in gradient {i}!")
        
        # Clip gradients
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config['training']['grad_clip_norm'])
        
        # Check if grad_norm is finite
        if not tf.math.is_finite(grad_norm):
            tf.print("Non-finite gradient norm detected:", grad_norm)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics - use only phoneme prediction for accuracy
        self.train_loss(total_loss)
        self.train_accuracy(phoneme_targets, phoneme_pred)
        
        return total_loss, grad_norm
        
    @tf.function
    def val_step(self, features, targets):
        """Single validation step."""
        # Unpack targets
        class_labels_onehot, ce_mask, new_class_signal = targets
        
        # Normalize features
        features = self.normalize_features(features)
        
        predictions = self.model(features, training=False)
        
        # Compute loss exactly like training
        n_classes = len(CMU_PHONEMES)
        
        # Create mask
        mask = tf.tile(ce_mask[:, :, tf.newaxis], [1, 1, n_classes])
        
        # Phoneme classification loss
        phoneme_pred = predictions[:, :, 0:n_classes] * mask
        phoneme_targets = class_labels_onehot[:, :, 0:n_classes]
        
        pred_loss = self.loss_fn(phoneme_targets, phoneme_pred)
        
        # New class signal loss
        new_class_pred = tf.math.sigmoid(predictions[:, :, -1])
        new_class_error = tf.reduce_mean(tf.math.square(new_class_pred - new_class_signal))
        
        total_loss = pred_loss + new_class_error
        
        # Update metrics
        self.val_loss(total_loss)
        self.val_accuracy(phoneme_targets, phoneme_pred)
        
        return total_loss
        
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Load data
        self.load_data()
        
        # Log model architecture info (deferred until model is built)
        self.log_model_info()
        
        # Training parameters
        epochs = self.config['training']['epochs']
        log_freq = self.config['training']['log_freq']
        save_freq = self.config['training']['save_freq']
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            start_time = time.time()
            
            # Reset metrics
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.val_loss.reset_state()
            self.val_accuracy.reset_state()
            
            # Training
            print("Training...")
            num_batches = 0
            batch_losses = []
            batch_grad_norms = []
            model_info_logged = False
            initial_val_logged = False
            
            for batch_idx, (features, targets) in enumerate(self.train_dataset):
                loss, grad_norm = self.train_step(features, targets)
                num_batches += 1
                
                # Log model info after first batch when model is built
                if not model_info_logged:
                    self.log_model_info()
                    model_info_logged = True
                
                # Log initial validation performance after first batch
                if not initial_val_logged:
                    print("Computing initial validation metrics...")
                    # Store current train metrics
                    temp_train_loss = self.train_loss.result()
                    temp_train_acc = self.train_accuracy.result()
                    
                    # Reset and compute initial validation
                    self.val_loss.reset_state()
                    self.val_accuracy.reset_state()
                    
                    for val_features, val_targets in self.val_dataset:
                        self.val_step(val_features, val_targets)
                        
                    initial_val_loss = float(self.val_loss.result())
                    initial_val_acc = float(self.val_accuracy.result())
                    
                    print(f"Initial validation - Loss: {initial_val_loss:.4f}, Acc: {initial_val_acc:.4f}")
                    
                    # Log initial metrics to wandb
                    self.log_to_wandb({
                        "val/loss": initial_val_loss,
                        "val/accuracy": initial_val_acc,
                        "epoch": 0
                    })
                    
                    # Reset validation metrics for this epoch
                    self.val_loss.reset_state()
                    self.val_accuracy.reset_state()
                    
                    initial_val_logged = True
                
                # Store batch metrics
                batch_losses.append(float(loss))
                batch_grad_norms.append(float(grad_norm))
                
                if (batch_idx + 1) % log_freq == 0:
                    current_lr = float(self.optimizer.learning_rate.numpy())
                    print(f"  Batch {batch_idx + 1}: loss={loss:.4f}, grad_norm={grad_norm:.4f}, lr={current_lr:.6f}")
                    
                    # Log batch-level metrics to wandb
                    self.log_to_wandb({
                        "batch/loss": float(loss),
                        "batch/grad_norm": float(grad_norm),
                        "batch/learning_rate": current_lr,
                        "batch/step": epoch * 1000 + batch_idx  # Approximate global step
                    })
            
            # Validation
            print("Validation...")
            for features, targets in self.val_dataset:
                self.val_step(features, targets)
            
            # Calculate epoch metrics
            epoch_time = time.time() - start_time
            train_loss_val = float(self.train_loss.result())
            train_acc_val = float(self.train_accuracy.result())
            val_loss_val = float(self.val_loss.result())
            val_acc_val = float(self.val_accuracy.result())
            current_lr = float(self.optimizer.learning_rate.numpy())
            
            # Print epoch results
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s:")
            print(f"  Train Loss: {train_loss_val:.4f}, Train Acc: {train_acc_val:.4f}")
            print(f"  Val Loss: {val_loss_val:.4f}, Val Acc: {val_acc_val:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Log epoch metrics to wandb
            epoch_metrics = {
                "epoch": epoch + 1,
                "train/loss": train_loss_val,
                "train/accuracy": train_acc_val,
                "val/loss": val_loss_val,
                "val/accuracy": val_acc_val,
                "train/learning_rate": current_lr,
                "train/epoch_time": epoch_time,
                "train/num_batches": num_batches,
                "train/avg_batch_loss": np.mean(batch_losses),
                "train/avg_grad_norm": np.mean(batch_grad_norms),
                "train/max_grad_norm": np.max(batch_grad_norms),
                "train/min_grad_norm": np.min(batch_grad_norms)
            }
            self.log_to_wandb(epoch_metrics)
            
            # Track best metrics
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                self.log_to_wandb({"best/val_loss": best_val_loss})
                print(f"  New best validation loss: {best_val_loss:.4f}")
                
            if val_acc_val > best_val_acc:
                best_val_acc = val_acc_val
                self.log_to_wandb({"best/val_accuracy": best_val_acc})
                print(f"  New best validation accuracy: {best_val_acc:.4f}")
                
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = self.checkpoint_manager.save()
                print(f"  Checkpoint saved: {checkpoint_path}")
                
                # Log checkpoint to wandb
                self.log_to_wandb({"checkpoint/epoch": epoch + 1, "checkpoint/path": checkpoint_path})
            
            # Update step counter
            self.checkpoint.step.assign_add(1)
            
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.4f}")
        
        # Log final metrics
        self.log_to_wandb({
            "final/best_val_loss": best_val_loss,
            "final/best_val_accuracy": best_val_acc,
            "final/total_epochs": epochs
        })
        
        # Save final model
        final_checkpoint = self.checkpoint_manager.save()
        print(f"Final checkpoint saved: {final_checkpoint}")
        
        # Save final checkpoint to wandb
        self.log_to_wandb({"final/checkpoint_path": final_checkpoint})
        wandb.save(final_checkpoint + "*")  # Save all checkpoint files

def create_default_config():
    """Create default configuration."""
    return {
        'wandb': {
            'enabled': True,
            'project': 'neural-ttt-decoding',
            'entity': None,  # Set to your wandb username/team if needed
            'mode': 'online',  # 'online', 'offline', or 'disabled'
            'save_code': True,
            'log_freq': 1  # Log every N epochs
        },
        'data': {
            'data_dir': 'sentences/',
            'use_spikepow': True,
            'use_tx1': True,
            'use_tx2': True,  # Now using all threshold crossing features
            'use_tx3': True,  # -5.5 x RMS threshold
            'use_tx4': True,  # -6.5 x RMS threshold
            'subsample_factor': 4,  # More aggressive subsampling to reduce sequence length
            'min_sentence_length': 10,
            'max_sentence_length': 200,  # Shorter sequences for initial training
            'train_split': 0.8,
            'max_files': None  # Use ALL available data files
        },
        'model': {
            'type': 'TTT_RNN',  # Test the fixed TTT implementation
            'units': 256,  # Increased from 128 for better capacity
            'weight_reg': 1e-6,  # Reduced regularization to allow more learning
            'act_reg': 1e-6,
            'subsample_factor': 1,
            'bidirectional': False,
            'dropout': 0.2,  # Moderate dropout
            'n_layers': 2,  # Increased from 1 for better representation
            'use_enhanced_ttt': False,  # Use basic TTT with the fix
            'ttt_config': {
                'inner_encoder': 'mlp_2',  # More expressive encoder
                'inner_iterations': 1,  # Keep simple for now
                'inner_lr': 0.01,  # Single learning rate for basic TTT
                'use_sgd': True,
                'decoder_ln': True,  # Enable layer norm for stability
                'sequence_length': 32  # Not used for basic TTT
            }
        },
        'training': {
            'output_dir': 'ttt_experiments',
            'epochs': 20,  # More epochs for better convergence
            'batch_size': 8,  # Increased batch size for more stable gradients
            'initial_lr': 3e-4,  # Significantly higher initial learning rate
            'final_lr': 1e-5,  # Better final learning rate
            'decay_steps': 1000,  # Longer decay schedule
            'decay_rate': 0.95,  # Gentler decay rate
            'use_cosine_decay': True,  # Use cosine decay for smoother learning
            'grad_clip_norm': 1.0,  # More relaxed gradient clipping
            'log_freq': 5,  # More frequent logging
            'save_freq': 5,  # Save every 5 epochs
            'weight_decay': 0  # Remove weight decay to avoid AdamW issues
        }
    }

def main():
    """Main function."""
    print("=== TTT-RNN Training on Neural Data ===")
    
    # Create default config
    config = create_default_config()
    
    try:
        # Create trainer and start training
        trainer = NeuralTrainer(config)
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Log error to wandb if initialized
        if wandb.run is not None:
            wandb.log({"error": str(e)})
        raise
    finally:
        # Clean up wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 