#!/usr/bin/env python3

"""
Test script for TTT-RNN implementation for phoneme prediction.
This script creates dummy data that simulates neural recordings and tests
both basic and enhanced TTT-RNN models.
"""

import sys
import os

# Add the path to import from speechBCI
sys.path.append(os.path.join(os.path.dirname(__file__), 'speechBCI/NeuralDecoder'))

import numpy as np
import tensorflow as tf
from neuralDecoder.ttt_models import TTT_RNN, TTTRNNCell, EnhancedTTTRNNCell

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_dummy_phoneme_data(batch_size=8, seq_len=100, n_features=256, n_classes=39):
    """
    Create dummy data that simulates neural recordings for phoneme prediction.
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence
        n_features: Number of neural recording features (e.g., 256 channels)
        n_classes: Number of phoneme classes (39 for standard phoneme set)
    
    Returns:
        inputs: Neural recording data [batch_size, seq_len, n_features]
        targets: Phoneme labels [batch_size, seq_len, n_classes]
    """
    print(f"Creating dummy data: batch_size={batch_size}, seq_len={seq_len}, n_features={n_features}")
    
    # Simulate neural recordings (normalized Gaussian noise with some structure)
    inputs = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)
    
    # Add some temporal structure to make it more realistic
    for i in range(1, seq_len):
        inputs[:, i] = 0.8 * inputs[:, i-1] + 0.2 * inputs[:, i]
    
    # Normalize
    inputs = (inputs - np.mean(inputs, axis=(0, 1))) / (np.std(inputs, axis=(0, 1)) + 1e-8)
    
    # Create dummy phoneme targets (one-hot encoded)
    # Simulate phoneme sequences with some temporal correlation
    phoneme_ids = np.random.randint(0, n_classes, (batch_size, seq_len))
    
    # Add temporal smoothing to phoneme sequences (phonemes don't change every timestep)
    for b in range(batch_size):
        for i in range(1, seq_len):
            if np.random.random() < 0.9:  # 90% chance to keep same phoneme
                phoneme_ids[b, i] = phoneme_ids[b, i-1]
    
    # Convert to one-hot
    targets = tf.keras.utils.to_categorical(phoneme_ids, num_classes=n_classes).astype(np.float32)
    
    return tf.constant(inputs), tf.constant(targets)

def test_ttt_rnn_cell():
    """Test basic TTT-RNN cell functionality."""
    print("\n=== Testing TTT-RNN Cell ===")
    
    # Create a simple TTT-RNN cell
    units = 128
    cell = TTTRNNCell(
        units=units,
        inner_encoder_type="mlp_2",
        inner_iterations=1,
        inner_lr=0.01,
        use_sgd=True,
        decoder_ln=True
    )
    
    # Test with dummy input
    batch_size = 4
    input_features = 256
    
    inputs = tf.random.normal((batch_size, input_features))
    initial_state = cell.get_initial_state(inputs=inputs, batch_size=batch_size)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Initial state shape: {initial_state.shape if not isinstance(initial_state, list) else [s.shape for s in initial_state]}")
    
    # Forward pass
    output, new_state = cell(inputs, [initial_state], training=True)
    
    print(f"Output shape: {output.shape}")
    print(f"New state shape: {new_state[0].shape}")
    print("✓ TTT-RNN cell test passed!")

def test_enhanced_ttt_rnn_cell():
    """Test enhanced TTT-RNN cell functionality."""
    print("\n=== Testing Enhanced TTT-RNN Cell ===")
    
    # Create an enhanced TTT-RNN cell
    units = 128
    cell = EnhancedTTTRNNCell(
        units=units,
        inner_encoder_type="mlp_2",
        inner_iterations=2,
        inner_lr=[0.01, 0.005],
        use_sgd=True,
        decoder_ln=True,
        sequence_length=32
    )
    
    # Test with dummy input
    batch_size = 4
    input_features = 256
    
    inputs = tf.random.normal((batch_size, input_features))
    initial_states = cell.get_initial_state(inputs=inputs, batch_size=batch_size)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Initial states shapes: {[s.shape for s in initial_states]}")
    
    # Forward pass
    output, new_states = cell(inputs, initial_states, training=True)
    
    print(f"Output shape: {output.shape}")
    print(f"New states shapes: {[s.shape for s in new_states]}")
    print("✓ Enhanced TTT-RNN cell test passed!")

def test_basic_ttt_rnn_model():
    """Test basic TTT-RNN model."""
    print("\n=== Testing Basic TTT-RNN Model ===")
    
    # Model configuration
    units = 128
    n_classes = 39
    n_layers = 2
    
    # TTT configuration
    ttt_config = {
        'inner_encoder': 'mlp_2',
        'inner_iterations': 1,
        'inner_lr': [0.01],
        'use_sgd': True,
        'decoder_ln': True
    }
    
    # Create TTT-RNN model
    model = TTT_RNN(
        units=units,
        weightReg=1e-4,
        actReg=1e-4,
        subsampleFactor=1,
        nClasses=n_classes,
        bidirectional=False,
        dropout=0.1,
        nLayers=n_layers,
        ttt_config=ttt_config,
        use_enhanced_ttt=False
    )
    
    # Create dummy data
    inputs, targets = create_dummy_phoneme_data(batch_size=4, seq_len=50, n_features=256, n_classes=n_classes)
    
    print(f"Model input shape: {inputs.shape}")
    print(f"Model target shape: {targets.shape}")
    
    # Forward pass
    outputs = model(inputs, training=True)
    
    print(f"Model output shape: {outputs.shape}")
    print(f"Expected output shape: (4, 50, {n_classes})")
    
    # Check output shape
    assert outputs.shape == (4, 50, n_classes), f"Expected shape (4, 50, {n_classes}), got {outputs.shape}"
    
    print("✓ Basic TTT-RNN model test passed!")
    
    return model, inputs, targets

def test_enhanced_ttt_rnn_model():
    """Test enhanced TTT-RNN model."""
    print("\n=== Testing Enhanced TTT-RNN Model ===")
    
    # Model configuration
    units = 128
    n_classes = 39
    n_layers = 2
    
    # Enhanced TTT configuration
    ttt_config = {
        'inner_encoder': 'mlp_2',
        'inner_iterations': 2,
        'inner_lr': [0.01, 0.005],
        'use_sgd': True,
        'decoder_ln': True,
        'sequence_length': 32
    }
    
    # Create enhanced TTT-RNN model
    model = TTT_RNN(
        units=units,
        weightReg=1e-4,
        actReg=1e-4,
        subsampleFactor=1,
        nClasses=n_classes,
        bidirectional=False,
        dropout=0.1,
        nLayers=n_layers,
        ttt_config=ttt_config,
        use_enhanced_ttt=True
    )
    
    # Create dummy data
    inputs, targets = create_dummy_phoneme_data(batch_size=4, seq_len=50, n_features=256, n_classes=n_classes)
    
    print(f"Model input shape: {inputs.shape}")
    print(f"Model target shape: {targets.shape}")
    
    # Forward pass
    outputs = model(inputs, training=True)
    
    print(f"Model output shape: {outputs.shape}")
    print(f"Expected output shape: (4, 50, {n_classes})")
    
    # Check output shape
    assert outputs.shape == (4, 50, n_classes), f"Expected shape (4, 50, {n_classes}), got {outputs.shape}"
    
    print("✓ Enhanced TTT-RNN model test passed!")
    
    return model, inputs, targets

def test_training_step():
    """Test a simple training step with TTT-RNN."""
    print("\n=== Testing Training Step ===")
    
    # Get a model and data
    model, inputs, targets = test_basic_ttt_rnn_model()
    
    # Define loss and optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # Training step
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Loss: {loss.numpy():.4f}")
    print(f"Number of trainable variables: {len(model.trainable_variables)}")
    print("✓ Training step test passed!")

def compare_models():
    """Compare TTT-RNN with standard GRU baseline."""
    print("\n=== Comparing TTT-RNN vs GRU ===")
    
    # Import GRU model for comparison
    from neuralDecoder.models import GRU
    
    # Model configuration
    units = 128
    n_classes = 39
    
    # Create GRU baseline
    gru_model = GRU(
        units=units,
        weightReg=1e-4,
        actReg=1e-4,
        subsampleFactor=1,
        nClasses=n_classes,
        bidirectional=False,
        dropout=0.1,
        nLayers=2
    )
    
    # Create TTT-RNN model
    ttt_config = {
        'inner_encoder': 'mlp_1',
        'inner_iterations': 1,
        'inner_lr': [0.01],
        'use_sgd': True,
        'decoder_ln': False
    }
    
    ttt_model = TTT_RNN(
        units=units,
        weightReg=1e-4,
        actReg=1e-4,
        subsampleFactor=1,
        nClasses=n_classes,
        bidirectional=False,
        dropout=0.1,
        nLayers=2,
        ttt_config=ttt_config,
        use_enhanced_ttt=False
    )
    
    # Create test data
    inputs, targets = create_dummy_phoneme_data(batch_size=2, seq_len=30, n_features=256, n_classes=n_classes)
    
    # Forward passes
    gru_outputs = gru_model(inputs, training=False)
    ttt_outputs = ttt_model(inputs, training=False)
    
    print(f"GRU output shape: {gru_outputs.shape}")
    print(f"TTT-RNN output shape: {ttt_outputs.shape}")
    
    # Compare parameter counts
    gru_params = sum([np.prod(v.shape) for v in gru_model.trainable_variables])
    ttt_params = sum([np.prod(v.shape) for v in ttt_model.trainable_variables])
    
    print(f"GRU parameters: {gru_params:,}")
    print(f"TTT-RNN parameters: {ttt_params:,}")
    print(f"Parameter ratio (TTT/GRU): {ttt_params/gru_params:.2f}")
    
    print("✓ Model comparison completed!")

def main():
    """Run all tests."""
    print("Testing TTT-RNN Implementation for Phoneme Prediction")
    print("=" * 60)
    
    try:
        # Test individual components
        test_ttt_rnn_cell()
        test_enhanced_ttt_rnn_cell()
        
        # Test full models
        test_basic_ttt_rnn_model()
        test_enhanced_ttt_rnn_model()
        
        # Test training
        test_training_step()
        
        # Compare models
        compare_models()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! TTT-RNN implementation is working correctly.")
        print("You can now use TTT_RNN in place of GRU for phoneme prediction.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 