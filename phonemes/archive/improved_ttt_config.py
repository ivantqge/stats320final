#!/usr/bin/env python3

"""
Improved TTT-RNN configurations for better performance.
Based on analysis of training behavior, these configs address specific issues:

1. Low accuracy (~40-45%)
2. Very low learning rates
3. Loss plateauing
4. Potential underfitting

"""

def get_aggressive_config():
    """
    Aggressive configuration for better learning.
    Higher learning rates, more capacity, longer training.
    """
    return {
        'data': {
            'data_dir': 'sentences/',
            'use_spikepow': True,
            'use_tx1': True,
            'use_tx2': False,
            'use_tx3': False,
            'use_tx4': False,
            'subsample_factor': 2,  # Less aggressive subsampling
            'min_sentence_length': 20,
            'max_sentence_length': 300,  # Longer sequences
            'train_split': 0.8,
            'max_files': 4  # More data
        },
        'model': {
            'type': 'TTT_RNN',
            'units': 384,  # Even larger model
            'weight_reg': 5e-7,  # Very little regularization
            'act_reg': 5e-7,
            'subsample_factor': 1,
            'bidirectional': False,
            'dropout': 0.1,  # Lower dropout
            'n_layers': 2,
            'use_enhanced_ttt': True,
            'ttt_config': {
                'inner_encoder': 'mlp_2',
                'inner_iterations': 3,  # More inner steps
                'inner_lr': [0.02, 0.01, 0.005],  # Aggressive inner learning
                'use_sgd': True,
                'decoder_ln': True,
                'sequence_length': 48  # Longer sequence buffer
            }
        },
        'training': {
            'output_dir': 'ttt_experiments',
            'epochs': 30,  # Much longer training
            'batch_size': 6,  # Balance memory and stability
            'initial_lr': 5e-4,  # Higher learning rate
            'final_lr': 5e-6,
            'decay_steps': 2000,
            'decay_rate': 0.96,
            'use_cosine_decay': True,
            'grad_clip_norm': 2.0,  # More relaxed clipping
            'log_freq': 5,
            'save_freq': 5,
            'weight_decay': 5e-5
        }
    }

def get_stable_config():
    """
    More conservative but stable configuration.
    Focus on steady learning with good regularization.
    """
    return {
        'data': {
            'data_dir': 'sentences/',
            'use_spikepow': True,
            'use_tx1': True,
            'use_tx2': False,
            'use_tx3': False,
            'use_tx4': False,
            'subsample_factor': 3,
            'min_sentence_length': 15,
            'max_sentence_length': 250,
            'train_split': 0.8,
            'max_files': 3
        },
        'model': {
            'type': 'TTT_RNN',
            'units': 256,
            'weight_reg': 2e-6,
            'act_reg': 2e-6,
            'subsample_factor': 1,
            'bidirectional': False,
            'dropout': 0.25,  # Moderate dropout
            'n_layers': 2,
            'use_enhanced_ttt': True,
            'ttt_config': {
                'inner_encoder': 'mlp_2',
                'inner_iterations': 2,
                'inner_lr': [0.01, 0.003],
                'use_sgd': True,
                'decoder_ln': True,
                'sequence_length': 32
            }
        },
        'training': {
            'output_dir': 'ttt_experiments',
            'epochs': 25,
            'batch_size': 8,
            'initial_lr': 2e-4,  # Moderate learning rate
            'final_lr': 2e-6,
            'decay_steps': 1500,
            'decay_rate': 0.94,
            'use_cosine_decay': True,
            'grad_clip_norm': 1.5,
            'log_freq': 5,
            'save_freq': 5,
            'weight_decay': 1e-4
        }
    }

def get_fast_config():
    """
    Configuration optimized for fast experimentation.
    Smaller model, shorter sequences, but good learning rates.
    """
    return {
        'data': {
            'data_dir': 'sentences/',
            'use_spikepow': True,
            'use_tx1': True,
            'use_tx2': False,
            'use_tx3': False,
            'use_tx4': False,
            'subsample_factor': 4,
            'min_sentence_length': 10,
            'max_sentence_length': 150,
            'train_split': 0.8,
            'max_files': 2
        },
        'model': {
            'type': 'TTT_RNN',
            'units': 128,
            'weight_reg': 1e-6,
            'act_reg': 1e-6,
            'subsample_factor': 1,
            'bidirectional': False,
            'dropout': 0.15,
            'n_layers': 1,
            'use_enhanced_ttt': False,  # Use basic TTT for speed
            'ttt_config': {
                'inner_encoder': 'mlp_1',
                'inner_iterations': 1,
                'inner_lr': [0.01],
                'use_sgd': True,
                'decoder_ln': False,
                'sequence_length': 16
            }
        },
        'training': {
            'output_dir': 'ttt_experiments',
            'epochs': 15,
            'batch_size': 12,  # Larger batches for speed
            'initial_lr': 3e-4,
            'final_lr': 1e-5,
            'decay_steps': 800,
            'decay_rate': 0.9,
            'use_cosine_decay': False,  # Exponential for simplicity
            'grad_clip_norm': 1.0,
            'log_freq': 10,
            'save_freq': 5,
            'weight_decay': 5e-5
        }
    }

def get_gru_baseline_config():
    """
    GRU baseline configuration for comparison.
    """
    return {
        'data': {
            'data_dir': 'sentences/',
            'use_spikepow': True,
            'use_tx1': True,
            'use_tx2': False,
            'use_tx3': False,
            'use_tx4': False,
            'subsample_factor': 3,
            'min_sentence_length': 15,
            'max_sentence_length': 250,
            'train_split': 0.8,
            'max_files': 3
        },
        'model': {
            'type': 'GRU',  # Standard GRU for baseline
            'units': 256,
            'weight_reg': 2e-6,
            'act_reg': 2e-6,
            'subsample_factor': 1,
            'bidirectional': False,
            'dropout': 0.25,
            'n_layers': 2,
            'use_enhanced_ttt': False,  # Not applicable to GRU
            'ttt_config': {}  # Not applicable to GRU
        },
        'training': {
            'output_dir': 'ttt_experiments',
            'epochs': 25,
            'batch_size': 8,
            'initial_lr': 3e-4,  # GRUs often work well with higher LR
            'final_lr': 1e-6,
            'decay_steps': 1500,
            'decay_rate': 0.94,
            'use_cosine_decay': True,
            'grad_clip_norm': 1.0,
            'log_freq': 5,
            'save_freq': 5,
            'weight_decay': 1e-4
        }
    }

# Configuration mapping
CONFIGS = {
    'aggressive': get_aggressive_config,
    'stable': get_stable_config,
    'fast': get_fast_config,
    'gru_baseline': get_gru_baseline_config
}

def get_config(config_name='stable'):
    """Get configuration by name."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]()

if __name__ == "__main__":
    import sys
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'stable'
    config = get_config(config_name)
    
    import yaml
    print(f"=== {config_name.upper()} CONFIG ===")
    print(yaml.dump(config, default_flow_style=False)) 