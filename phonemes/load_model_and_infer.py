#!/usr/bin/env python3
"""
Script to load the TTT-RNN model and run inference to get actual results.
This script is designed to work with the actual trained model and data.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import argparse

# Add the speechBCI path
sys.path.append('speechBCI/NeuralDecoder')
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from omegaconf import OmegaConf

def load_and_infer(experiment_dir, data_config):
    """
    Load the trained model and run inference.
    
    Args:
        experiment_dir: Path to experiment directory with checkpoints
        data_config: Configuration for data paths and dataset info
    """
    
    experiment_path = Path(experiment_dir)
    
    # Load the training config
    with open(experiment_path / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    print(f"  Model type: {config['model']['type']}")
    print(f"  Units: {config['model']['units']}")
    print(f"  TTT config: {config['model']['ttt_config']}")
    
    # Convert to format expected by NeuralSequenceDecoder
    args = {
        'mode': 'infer',
        'outputDir': str(experiment_path / 'checkpoints'),
        'loadDir': str(experiment_path / 'checkpoints'),
        'loadCheckpointIdx': None,  # Use latest checkpoint
        'seed': 42,
        
        # Model configuration from training config
        'model': {
            'type': config['model']['type'],
            'nUnits': config['model']['units'],
            'weightReg': config['model']['weight_reg'],
            'actReg': config['model']['act_reg'],
            'subsampleFactor': config['model']['subsample_factor'],
            'bidirectional': config['model']['bidirectional'],
            'dropout': config['model']['dropout'],
            'nLayers': config['model']['n_layers'],
            'ttt_config': config['model']['ttt_config'],
            'use_enhanced_ttt': config['model']['use_enhanced_ttt']
        },
        
        # Dataset configuration - YOU NEED TO FILL THESE IN
        'dataset': {
            'name': data_config.get('dataset_name', 'SpeechDataset'),
            'dataDir': data_config['data_dirs'],
            'sessions': data_config['sessions'],
            'nInputFeatures': data_config['n_input_features'],
            'nClasses': data_config['n_classes'],
            'maxSeqElements': config['data']['max_sentence_length'],
            'bufferSize': 1000,
            'datasetToLayerMap': [0] * len(data_config['sessions']),
            'datasetProbabilityVal': [1.0] * len(data_config['sessions']),
            'syntheticMixingRate': 0,
            'syntheticDataDir': None
        },
        
        # Other required parameters
        'batchSize': config['training']['batch_size'],
        'lossType': 'ctc',  # or 'ce' depending on your setup
        'normLayer': True,
        'trainableBackend': True,
        'trainableInput': True,
        'smoothInputs': False,
        'batchesPerVal': 100,
        
        # Learning rate parameters (not used in inference but required)
        'learnRateStart': config['training']['initial_lr'],
        'learnRateEnd': config['training']['final_lr'],
        'learnRatePower': 1.0,
        'nBatchesToTrain': 1000,
        'warmUpSteps': 0,
        'gradClipValue': config['training']['grad_clip_norm']
    }
    
    args = OmegaConf.create(args)
    
    try:
        print("Initializing NeuralSequenceDecoder...")
        decoder = NeuralSequenceDecoder(args)
        print("Model loaded successfully!")
        
        print("Running inference...")
        results = decoder.inference(returnData=False)
        
        print("Inference completed!")
        print(f"Results keys: {list(results.keys())}")
        
        if 'cer' in results:
            print(f"Character Error Rate: {results['cer']*100:.2f}%")
        
        if 'editDistances' in results and 'trueSeqLengths' in results:
            cer = np.sum(results['editDistances']) / np.sum(results['trueSeqLengths'])
            print(f"Computed CER: {cer*100:.2f}%")
        
        # Save results
        output_file = experiment_path / 'inference_results.npz'
        np.savez(output_file, **results)
        print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error during model loading or inference: {e}")
        print("This likely means you need to configure the data paths correctly.")
        print("Please update the data_config in the main function below.")
        raise

def main():
    parser = argparse.ArgumentParser(description='Load TTT-RNN model and run inference')
    parser.add_argument('--experiment_dir', 
                       default='ttt_experiments/basic_ttt_20250609_020007',
                       help='Path to experiment directory')
    
    args = parser.parse_args()
    
    # DATA CONFIGURATION - YOU NEED TO FILL THESE IN WITH YOUR ACTUAL PATHS
    data_config = {
        'dataset_name': 'YourDatasetClassName',  # Replace with actual dataset class
        'data_dirs': [
            '/path/to/your/data/directory1',     # Replace with actual data paths
            '/path/to/your/data/directory2'      # Add more as needed
        ],
        'sessions': [
            'session_name_1',                    # Replace with actual session names
            'session_name_2'                     # Add more as needed
        ],
        'n_input_features': 256,                 # Replace with actual number of input features
        'n_classes': 40                          # Replace with actual number of phoneme classes
    }
    
    print("=" * 60)
    print("TTT-RNN Model Loading and Inference Script")
    print("=" * 60)
    print()
    print("IMPORTANT: Before running this script, you need to:")
    print("1. Update the data_config dictionary above with your actual:")
    print("   - Dataset class name")
    print("   - Data directory paths") 
    print("   - Session names")
    print("   - Number of input features")
    print("   - Number of phoneme classes")
    print()
    print("2. Make sure your data is in the expected format for the")
    print("   NeuralSequenceDecoder class")
    print()
    
    # Uncomment the line below once you've configured the paths
    # results = load_and_infer(args.experiment_dir, data_config)
    
    print("Please configure the data_config and uncomment the load_and_infer call.")

if __name__ == '__main__':
    main() 