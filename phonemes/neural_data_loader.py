#!/usr/bin/env python3

"""
Neural Data Loader for Sentences Dataset
Loads MATLAB files containing neural recordings and phoneme targets for TTT-RNN training.
"""

import os
import glob
import numpy as np
import scipy.io
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import h5py
import re
from collections import defaultdict

# CMU Phoneme set (39 phonemes) in the order specified in the data description
CMU_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Create phoneme to index mapping
PHONEME_TO_IDX = {phoneme: idx for idx, phoneme in enumerate(CMU_PHONEMES)}
IDX_TO_PHONEME = {idx: phoneme for phoneme, idx in PHONEME_TO_IDX.items()}

class NeuralDataLoader:
    """
    Loads and processes neural data from MATLAB files for phoneme prediction.
    """
    
    def __init__(self, 
                 data_dir: str = "sentences/",
                 use_spikepow: bool = True,
                 use_tx1: bool = True,
                 use_tx2: bool = False,
                 use_tx3: bool = False,
                 use_tx4: bool = False,
                 subsample_factor: int = 1,
                 min_sentence_length: int = 10,
                 max_sentence_length: int = 1000):
        """
        Initialize the neural data loader.
        
        Args:
            data_dir: Directory containing .mat files
            use_spikepow: Whether to include spike power features
            use_tx1-tx4: Whether to include threshold crossing features
            subsample_factor: Factor to subsample temporal data
            min_sentence_length: Minimum sentence length in time bins
            max_sentence_length: Maximum sentence length in time bins
        """
        self.data_dir = data_dir
        self.use_spikepow = use_spikepow
        self.use_tx1 = use_tx1
        self.use_tx2 = use_tx2
        self.use_tx3 = use_tx3
        self.use_tx4 = use_tx4
        self.subsample_factor = subsample_factor
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        
        # Calculate number of features
        self.n_features = 0
        if use_spikepow: self.n_features += 256
        if use_tx1: self.n_features += 256
        if use_tx2: self.n_features += 256
        if use_tx3: self.n_features += 256
        if use_tx4: self.n_features += 256
        
        print(f"Neural data loader initialized with {self.n_features} features")
        
    def load_mat_file(self, filepath: str) -> Dict:
        """
        Load a MATLAB file and return its contents.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary containing the loaded data
        """
        print(f"Loading {filepath}...")
        
        try:
            # Try loading with scipy.io first (for older MATLAB files)
            data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
            return data
        except:
            # Try with h5py for newer MATLAB files
            try:
                data = {}
                with h5py.File(filepath, 'r') as f:
                    for key in f.keys():
                        if not key.startswith('__'):
                            data[key] = np.array(f[key])
                return data
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return {}
                
    def get_mat_files(self) -> List[str]:
        """Get list of all .mat files in the data directory."""
        pattern = os.path.join(self.data_dir, "*.mat")
        files = sorted(glob.glob(pattern))
        print(f"Found {len(files)} .mat files")
        return files
    
    def extract_features(self, data: Dict) -> np.ndarray:
        """
        Extract neural features from loaded data.
        
        Args:
            data: Dictionary containing loaded MATLAB data
            
        Returns:
            features: Array of shape [time_steps, n_features]
        """
        feature_list = []
        
        if self.use_spikepow and 'spikePow' in data:
            spikepow = data['spikePow']
            if spikepow.ndim == 1:
                spikepow = spikepow.reshape(-1, 1)
            feature_list.append(spikepow)
            
        if self.use_tx1 and 'tx1' in data:
            tx1 = data['tx1']
            if tx1.ndim == 1:
                tx1 = tx1.reshape(-1, 1)
            feature_list.append(tx1)
            
        if self.use_tx2 and 'tx2' in data:
            tx2 = data['tx2']
            if tx2.ndim == 1:
                tx2 = tx2.reshape(-1, 1)
            feature_list.append(tx2)
            
        if self.use_tx3 and 'tx3' in data:
            tx3 = data['tx3']
            if tx3.ndim == 1:
                tx3 = tx3.reshape(-1, 1)
            feature_list.append(tx3)
            
        if self.use_tx4 and 'tx4' in data:
            tx4 = data['tx4']
            if tx4.ndim == 1:
                tx4 = tx4.reshape(-1, 1)
            feature_list.append(tx4)
            
        if feature_list:
            features = np.concatenate(feature_list, axis=1)
            
            # Apply subsampling if specified
            if self.subsample_factor > 1:
                features = features[::self.subsample_factor, :]
                
            return features
        else:
            return np.array([])
    
    def text_to_phonemes(self, text):
        """
        Convert text to phoneme sequence.
        This is a placeholder - in practice you'd use a proper phoneme converter.
        """
        words = text.lower().split()
        phonemes = []
        
        # Very simple mapping - just for testing
        word_to_phonemes = {
            'the': ['DH', 'AH'],
            'quick': ['K', 'W', 'IH', 'K'],
            'brown': ['B', 'R', 'AW', 'N'],
            'fox': ['F', 'AA', 'K', 'S'],
            'jumps': ['JH', 'AH', 'M', 'P', 'S'],
            'over': ['OW', 'V', 'ER'],
            'lazy': ['L', 'EY', 'Z', 'IY'],
            'dog': ['D', 'AA', 'G'],
            'hello': ['HH', 'EH', 'L', 'OW'],
            'world': ['W', 'ER', 'L', 'D'],
            'test': ['T', 'EH', 'S', 'T'],
            'sentence': ['S', 'EH', 'N', 'T', 'AH', 'N', 'S'],
            'speaking': ['S', 'P', 'IY', 'K', 'IH', 'NG'],
            'neural': ['N', 'UH', 'R', 'AH', 'L'],
            'decoder': ['D', 'IY', 'K', 'OW', 'D', 'ER']
        }
        
        for word in words:
            if word in word_to_phonemes:
                phonemes.extend(word_to_phonemes[word])
            else:
                # Default mapping for unknown words
                phonemes.extend(['AH', 'N', 'OW', 'N'])  # "unknown"
        
        # Convert to indices (1-based to match original implementation)
        phoneme_indices = []
        for phoneme in phonemes:
            if phoneme in CMU_PHONEMES:
                phoneme_indices.append(CMU_PHONEMES.index(phoneme) + 1)  # 1-based indexing
            else:
                phoneme_indices.append(1)  # Default to first phoneme
                
        return phoneme_indices
    
    def create_targets_and_masks(self, phoneme_indices, sequence_length):
        """
        Create one-hot encoded targets and masks matching the original implementation.
        
        Args:
            phoneme_indices: List of phoneme indices (1-based)
            sequence_length: Length of the neural sequence
            
        Returns:
            classLabelsOneHot: One-hot encoded labels [time, nClasses+1]
            ceMask: Mask for valid timesteps [time]
            newClassSignal: Signal for new class detection [time]
        """
        n_classes = len(CMU_PHONEMES)
        
        # Create targets - allocate phonemes across the sequence
        targets = np.zeros(sequence_length, dtype=np.int32)
        ce_mask = np.zeros(sequence_length, dtype=np.float32)
        new_class_signal = np.zeros(sequence_length, dtype=np.float32)
        
        if len(phoneme_indices) > 0:
            # Spread phonemes across the sequence
            phoneme_length = sequence_length // len(phoneme_indices)
            remainder = sequence_length % len(phoneme_indices)
            
            pos = 0
            for i, phoneme_idx in enumerate(phoneme_indices):
                # Add extra time step to first 'remainder' phonemes
                length = phoneme_length + (1 if i < remainder else 0)
                
                # Fill the segment with this phoneme
                targets[pos:pos+length] = phoneme_idx
                ce_mask[pos:pos+length] = 1.0  # Valid time steps
                
                # Mark new class signal at the beginning of each phoneme
                if pos < sequence_length:
                    new_class_signal[pos] = 1.0
                    
                pos += length
                if pos >= sequence_length:
                    break
        
        # Convert to one-hot encoding with extra dimension for new class signal
        # Shape: [time, nClasses + 1] where last dim is for new class signal
        class_labels_onehot = np.zeros((sequence_length, n_classes + 1), dtype=np.float32)
        
        for t in range(sequence_length):
            if ce_mask[t] > 0:  # Valid time step
                if targets[t] > 0 and targets[t] <= n_classes:
                    class_labels_onehot[t, targets[t] - 1] = 1.0  # Convert back to 0-based for one-hot
                    
        # The last dimension gets the new class signal
        class_labels_onehot[:, -1] = new_class_signal
        
        return class_labels_onehot, ce_mask, new_class_signal
    
    def process_single_file(self, filepath: str) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process a single .mat file and extract features and targets.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Tuple of (features_list, targets_list) for each sentence
        """
        data = self.load_mat_file(filepath)
        if not data:
            return [], []
            
        features_list = []
        targets_list = []
        
        # Check if we have sentence-level data
        if 'sentencesFeatures' in data and 'sentences' in data:
            sentences_features = data['sentencesFeatures']
            sentences = data['sentences']
            
            # Handle different data formats
            if hasattr(sentences_features, '__len__'):
                n_sentences = len(sentences_features)
            else:
                n_sentences = 1
                sentences_features = [sentences_features]
                sentences = [sentences]
            
            print(f"Processing {n_sentences} sentences from {os.path.basename(filepath)}")
            
            for i in range(n_sentences):
                try:
                    # Extract features for this sentence
                    if hasattr(sentences_features[i], 'shape'):
                        sentence_features = sentences_features[i]
                    else:
                        sentence_features = np.array(sentences_features[i])
                    
                    # The description says first 256 are tx1, last 256 are spikePow
                    if sentence_features.shape[-1] >= 512:
                        if self.use_tx1 and self.use_spikepow:
                            # Use both tx1 and spikePow as described
                            features = sentence_features
                        elif self.use_tx1:
                            # Use only tx1 features
                            features = sentence_features[:, :256]
                        elif self.use_spikepow:
                            # Use only spikePow features
                            features = sentence_features[:, 256:512]
                        else:
                            continue
                    else:
                        features = sentence_features
                    
                    # Apply subsampling
                    if self.subsample_factor > 1:
                        features = features[::self.subsample_factor, :]
                    
                    # Check sentence length constraints
                    if features.shape[0] < self.min_sentence_length or features.shape[0] > self.max_sentence_length:
                        continue
                    
                    # Extract target sentence
                    if hasattr(sentences, '__len__') and i < len(sentences):
                        sentence_text = str(sentences[i])
                    else:
                        sentence_text = str(sentences)
                    
                    # Convert to phonemes and then to indices
                    phoneme_indices = self.text_to_phonemes(sentence_text)
                    
                    # Create targets and masks
                    class_labels_onehot, ce_mask, new_class_signal = self.create_targets_and_masks(phoneme_indices, features.shape[0])
                    
                    features_list.append(features.astype(np.float32))
                    # Store both targets and mask - we'll need the mask for loss computation
                    targets_list.append({
                        'class_labels_onehot': class_labels_onehot,
                        'ce_mask': ce_mask,
                        'new_class_signal': new_class_signal
                    })
                    
                except Exception as e:
                    print(f"Error processing sentence {i} in {filepath}: {e}")
                    continue
        
        print(f"Successfully processed {len(features_list)} sentences from {os.path.basename(filepath)}")
        return features_list, targets_list
    
    def load_all_data(self, max_files: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load data from all .mat files.
        
        Args:
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            Tuple of (all_features, all_targets)
        """
        mat_files = self.get_mat_files()
        
        if max_files is not None:
            mat_files = mat_files[:max_files]
            
        all_features = []
        all_targets = []
        
        for filepath in mat_files:
            features_list, targets_list = self.process_single_file(filepath)
            all_features.extend(features_list)
            all_targets.extend(targets_list)
            
        print(f"Loaded {len(all_features)} total sentences from {len(mat_files)} files")
        return all_features, all_targets
    
    def create_tensorflow_dataset(self, 
                                features_list: List[np.ndarray], 
                                targets_list: List[Dict],
                                batch_size: int = 32,
                                shuffle: bool = True,
                                buffer_size: int = 1000) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from features and targets.
        
        Args:
            features_list: List of feature arrays
            targets_list: List of target dictionaries
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            buffer_size: Buffer size for shuffling
            
        Returns:
            TensorFlow dataset
        """
        # Convert to generator function to avoid ragged tensor issues
        def data_generator():
            for features, targets in zip(features_list, targets_list):
                yield features.astype(np.float32), (
                    targets['class_labels_onehot'].astype(np.float32),
                    targets['ce_mask'].astype(np.float32),
                    targets['new_class_signal'].astype(np.float32)
                )
        
        # Get output shapes and types
        n_features = features_list[0].shape[1]
        n_classes = len(CMU_PHONEMES)
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(None, n_classes + 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),  # CE mask is 1D
                    tf.TensorSpec(shape=(None,), dtype=tf.float32)   # New class signal is 1D
                )
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
            
        # Pad and batch
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, n_features], ([None, n_classes + 1], [None], [None])),
            padding_values=(0.0, (0.0, 0.0, 0.0))
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def main():
    """Test the data loader."""
    print("Testing Neural Data Loader...")
    
    # Initialize loader
    loader = NeuralDataLoader(
        data_dir="sentences/",
        use_spikepow=True,
        use_tx1=True,
        subsample_factor=1,
        min_sentence_length=10,
        max_sentence_length=500
    )
    
    # Load a small sample of data
    print("\nLoading data from first file...")
    mat_files = loader.get_mat_files()
    if mat_files:
        features_list, targets_list = loader.process_single_file(mat_files[0])
        
        if features_list:
            print(f"Sample shapes:")
            print(f"Features: {features_list[0].shape}")
            print(f"Targets: {targets_list[0]['class_labels_onehot'].shape}")
            print(f"Feature range: [{features_list[0].min():.3f}, {features_list[0].max():.3f}]")
            print(f"Target range: [{targets_list[0]['class_labels_onehot'].min()}, {targets_list[0]['class_labels_onehot'].max()}]")
            
            # Create a small dataset
            dataset = loader.create_tensorflow_dataset(
                features_list[:5], targets_list[:5], batch_size=2
            )
            
            print(f"\nTesting TensorFlow dataset...")
            for batch_idx, (features_batch, (targets_batch, ce_mask_batch, new_class_signal_batch)) in enumerate(dataset.take(2)):
                print(f"Batch {batch_idx}:")
                print(f"  Features batch shape: {features_batch.shape}")
                print(f"  Targets batch shape: {targets_batch.shape}")
                print(f"  CE Mask batch shape: {ce_mask_batch.shape}")
                print(f"  New Class Signal batch shape: {new_class_signal_batch.shape}")
                
        else:
            print("No valid sentences found in the first file.")
    else:
        print("No .mat files found!")

if __name__ == "__main__":
    main() 