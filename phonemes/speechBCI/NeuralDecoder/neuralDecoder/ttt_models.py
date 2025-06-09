import tensorflow as tf
from tensorflow.keras import Model
import numpy as np


class EnhancedTTTRNNCell(tf.keras.layers.Layer):
    """
    Enhanced TTT-based RNN cell with proper inner optimization loop.
    This implementation more closely follows the MTTT paper's approach.
    """
    
    def __init__(self, 
                 units,
                 inner_encoder_type="mlp_2",
                 inner_iterations=1,
                 inner_lr=[0.01],
                 use_sgd=True,
                 decoder_ln=True,
                 sequence_length=32,  # Length of sequence to maintain for TTT
                 **kwargs):
        super(EnhancedTTTRNNCell, self).__init__(**kwargs)
        
        self.units = units
        self.inner_encoder_type = inner_encoder_type
        self.inner_iterations = inner_iterations
        self.inner_lr = inner_lr if isinstance(inner_lr, list) else [inner_lr]
        self.use_sgd = use_sgd
        self.decoder_ln = decoder_ln 
        self.sequence_length = sequence_length
        self.state_size = [units, tf.TensorShape([sequence_length, units])]  # Hidden state + sequence buffer
        
        # Build the TTT components
        self._build_ttt_components()
        
    def _build_ttt_components(self):
        """Build the TTT encoder and decoder components."""
        # Phi network (transforms input to inner representation)
        self.phi_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=True,
            name="phi_projection"
        )
        
        # Psi network (transforms input for final output)
        self.psi_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=True,
            name="psi_projection"
        )
        
        # Inner encoder (adapts during inference)
        if self.inner_encoder_type == "mlp_1":
            self.inner_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(self.units, activation=None, use_bias=True, name="inner_dense_0")
            ], name="inner_encoder")
        elif self.inner_encoder_type == "mlp_2":
            self.inner_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(self.units * 4, activation='gelu', use_bias=True, name="inner_dense_0"),
                tf.keras.layers.Dense(self.units, activation=None, use_bias=True, name="inner_dense_1")
            ], name="inner_encoder")
        else:
            raise ValueError(f"Unknown inner encoder type: {self.inner_encoder_type}")
            
        # Decoder g (reconstructs from inner representation)
        self.g_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=False,
            name="g_decoder"
        )
        
        # Output h (final transformation)
        self.h_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=False,
            name="h_output"
        )
        
        # Optional layer normalization
        if self.decoder_ln:
            self.layer_norm = tf.keras.layers.LayerNormalization(name="decoder_ln")
        
    def build(self, input_shape):
        """Build the layer."""
        super(EnhancedTTTRNNCell, self).build(input_shape)
        
        # Create input projection layer for when input features don't match units
        self.input_projection = tf.keras.layers.Dense(self.units, name="input_projection")
        
        # Initialize bias parameters
        self.g_bias = self.add_weight(
            name="g_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        
        self.h_bias = self.add_weight(
            name="h_bias", 
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial hidden state and sequence buffer."""
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        elif batch_size is None:
            raise ValueError("Either inputs or batch_size must be provided")
            
        if dtype is None and inputs is not None:
            dtype = inputs.dtype
        elif dtype is None:
            dtype = tf.float32
            
        # Initial hidden state
        initial_hidden = tf.zeros((batch_size, self.units), dtype=dtype)
        
        # Initial sequence buffer for TTT
        initial_sequence = tf.zeros((batch_size, self.sequence_length, self.units), dtype=dtype)
        
        return [initial_hidden, initial_sequence]
    
    def call(self, inputs, states, training=False):
        """
        Forward pass through Enhanced TTT-RNN cell.
        
        Args:
            inputs: Input tensor [batch_size, input_features]
            states: [Previous hidden state [batch_size, units], 
                    Sequence buffer [batch_size, sequence_length, units]]
            training: Whether in training mode
            
        Returns:
            output: Cell output [batch_size, units]
            new_states: Updated [hidden state, sequence buffer]
        """
        prev_hidden, sequence_buffer = states
        batch_size = tf.shape(inputs)[0]
        
        # Transform input to same dimension as hidden state using the pre-built projection layer
        if inputs.shape[-1] != self.units:
            projected_input = self.input_projection(inputs)
        else:
            projected_input = inputs
        
        # Update sequence buffer with new input
        # Shift buffer and add new input
        new_sequence_buffer = tf.concat([
            sequence_buffer[:, 1:, :],  # Remove first element
            tf.expand_dims(projected_input, axis=1)  # Add new input at the end
        ], axis=1)
        
        # Apply TTT mechanism using the sequence buffer
        ttt_output, inner_losses = self._apply_enhanced_ttt(
            new_sequence_buffer, 
            prev_hidden, 
            training=training
        )
        
        # The TTT output becomes our new hidden state
        new_hidden = ttt_output
        
        return new_hidden, [new_hidden, new_sequence_buffer]
    
    def _apply_enhanced_ttt(self, sequence_buffer, prev_hidden, training=False):
        """
        Apply enhanced TTT mechanism with actual inner optimization.
        
        Args:
            sequence_buffer: Sequence buffer [batch_size, seq_len, units]
            prev_hidden: Previous hidden state [batch_size, units]
            training: Whether in training mode
            
        Returns:
            output: TTT output [batch_size, units]
            inner_losses: List of inner optimization losses
        """
        batch_size = tf.shape(sequence_buffer)[0]
        seq_len = tf.shape(sequence_buffer)[1]
        
        # Create targets for self-supervised learning
        # We'll use next-token prediction as the objective
        inputs_seq = sequence_buffer[:, :-1, :]  # All but last
        targets_seq = sequence_buffer[:, 1:, :]   # All but first
        
        # Flatten for processing
        inputs_flat = tf.reshape(inputs_seq, [-1, self.units])
        targets_flat = tf.reshape(targets_seq, [-1, self.units])
        
        # Transform inputs through phi network
        phi_output = self.phi_dense(inputs_flat)
        
        inner_losses = []
        
        # Ensure inner encoder is built by calling it once
        if not self.inner_encoder.built:
            _ = self.inner_encoder(phi_output, training=training)
        
        # Save original inner encoder parameters for restoration
        original_weights = []
        for var in self.inner_encoder.trainable_variables:
            original_weights.append(tf.Variable(var))
        
        try:
            # Store the final reconstruction for gradient flow
            final_reconstruction = None
            
            for iteration in range(self.inner_iterations):
                with tf.GradientTape() as tape:
                    # Forward pass through inner encoder with current weights
                    inner_transformed = self.inner_encoder(phi_output, training=training)
                    
                    # Decode back to target space
                    reconstructed = self.g_dense(inner_transformed) + self.g_bias
                    
                    # Apply layer normalization if enabled
                    if self.decoder_ln:
                        reconstructed = self.layer_norm(reconstructed, training=training)
                    
                    # Compute reconstruction loss (self-supervised objective)
                    inner_loss = tf.reduce_mean(tf.square(reconstructed - targets_flat))
                    inner_losses.append(inner_loss)
                
                # Compute gradients with respect to inner encoder parameters
                gradients = tape.gradient(inner_loss, self.inner_encoder.trainable_variables)
                
                # Apply gradient updates (use appropriate learning rate for iteration)
                current_lr = self.inner_lr[min(iteration, len(self.inner_lr)-1)] if isinstance(self.inner_lr, list) else self.inner_lr
                for var, grad in zip(self.inner_encoder.trainable_variables, gradients):
                    if grad is not None:
                        var.assign_sub(current_lr * grad)
                
                # Keep the final reconstruction for the output pathway
                final_reconstruction = reconstructed
                    
            # Generate final output using psi pathway with updated encoder
            # Use the most recent input for final prediction  
            latest_input = sequence_buffer[:, -1, :]  # [batch_size, units]
            psi_output = self.psi_dense(latest_input)
            
            # Combine the inner reconstruction with psi pathway for final output
            # This ensures all components contribute to gradients
            inner_contribution = tf.reduce_mean(tf.reshape(final_reconstruction, [batch_size, -1, self.units]), axis=1)
            inner_processed = self.inner_encoder(psi_output, training=training)
            final_output = self.h_dense(inner_processed + inner_contribution) + self.h_bias
            
        finally:
            # Restore original parameters for next time step
            for original_var, current_var in zip(original_weights, self.inner_encoder.trainable_variables):
                current_var.assign(original_var)
        
        return final_output, inner_losses


class TTTRNNCell(tf.keras.layers.Layer):
    """
    TTT-based RNN cell that adapts the TTT mechanism for sequential processing.
    This cell maintains both hidden states and inner model parameters across time steps.
    """
    
    def __init__(self, 
                 units,
                 inner_encoder_type="mlp_2",
                 inner_iterations=1,
                 inner_lr=0.01,
                 use_sgd=True,
                 decoder_ln=True,
                 **kwargs):
        super(TTTRNNCell, self).__init__(**kwargs)
        
        self.units = units
        self.inner_encoder_type = inner_encoder_type
        self.inner_iterations = inner_iterations
        self.inner_lr = inner_lr
        self.use_sgd = use_sgd
        self.decoder_ln = decoder_ln
        self.state_size = units
        
        # Build the TTT components
        self._build_ttt_components()
        
    def _build_ttt_components(self):
        """Build the TTT encoder and decoder components."""
        # Phi network (transforms input to inner representation)
        self.phi_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=True,
            name="phi_projection"
        )
        
        # Psi network (transforms input for final output)
        self.psi_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=True,
            name="psi_projection"
        )
        
        # Inner encoder (adapts during inference)
        if self.inner_encoder_type == "mlp_1":
            self.inner_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(self.units, activation=None, use_bias=True, name="inner_dense_0")
            ], name="inner_encoder")
        elif self.inner_encoder_type == "mlp_2":
            self.inner_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(self.units * 4, activation='gelu', use_bias=True, name="inner_dense_0"),
                tf.keras.layers.Dense(self.units, activation=None, use_bias=True, name="inner_dense_1")
            ], name="inner_encoder")
        else:
            raise ValueError(f"Unknown inner encoder type: {self.inner_encoder_type}")
            
        # Decoder g (reconstructs from inner representation)
        self.g_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=False,
            name="g_decoder"
        )
        
        # Output h (final transformation)
        self.h_dense = tf.keras.layers.Dense(
            self.units,
            activation=None,
            use_bias=False,
            name="h_output"
        )
        
        # Optional layer normalization
        if self.decoder_ln:
            self.layer_norm = tf.keras.layers.LayerNormalization(name="decoder_ln")
        
    def build(self, input_shape):
        """Build the layer."""
        super(TTTRNNCell, self).build(input_shape)
        
        # Initialize bias parameters
        self.g_bias = self.add_weight(
            name="g_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        
        self.h_bias = self.add_weight(
            name="h_bias", 
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial hidden state."""
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        elif batch_size is None:
            raise ValueError("Either inputs or batch_size must be provided")
            
        if dtype is None and inputs is not None:
            dtype = inputs.dtype
        elif dtype is None:
            dtype = tf.float32
            
        # Initial hidden state
        initial_hidden = tf.zeros((batch_size, self.units), dtype=dtype)
        
        # Initial inner model parameters (we'll store them as state for simplicity)
        # In practice, you might want to handle this differently
        return initial_hidden
    
    def call(self, inputs, states, training=False):
        """
        Forward pass through TTT-RNN cell.
        
        Args:
            inputs: Input tensor [batch_size, input_features]
            states: Previous hidden state [batch_size, units]
            training: Whether in training mode
            
        Returns:
            output: Cell output [batch_size, units]
            new_states: Updated hidden state [batch_size, units]
        """
        prev_hidden = states[0]
        
        # Combine current input with previous hidden state
        # This is the key difference from standard TTT - we use sequential context
        combined_input = tf.concat([inputs, prev_hidden], axis=-1)
        
        # For TTT, we need to create a sequence for the inner loop
        # Here we'll use a simple approach - treat the combined input as our "sequence"
        # In a more sophisticated implementation, you might maintain a sliding window
        sequence_input = combined_input
        
        # Apply TTT mechanism
        ttt_output, inner_losses = self._apply_ttt(sequence_input, training=training)
        
        # The TTT output becomes our new hidden state
        new_hidden = ttt_output
        
        return new_hidden, [new_hidden]
    
    def _apply_ttt(self, sequence_input, training=False):
        """
        Apply TTT mechanism to the input sequence with actual gradient steps.
        
        Args:
            sequence_input: Input for TTT processing [batch_size, features]
            training: Whether in training mode
            
        Returns:
            output: TTT output [batch_size, units]
            inner_losses: List of inner optimization losses
        """
        batch_size = tf.shape(sequence_input)[0]
        
        # Transform input through phi network
        phi_output = self.phi_dense(sequence_input)
        
        # For the inner loop, we need to create a self-supervised objective
        # We'll use reconstruction as the objective
        target = phi_output  # Reconstruct the phi-transformed input
        
        inner_losses = []
        
        # Ensure inner encoder is built by calling it once
        if not self.inner_encoder.built:
            _ = self.inner_encoder(phi_output, training=training)
        
        # Save original inner encoder parameters for restoration
        original_weights = []
        for var in self.inner_encoder.trainable_variables:
            original_weights.append(tf.Variable(var))
        
        try:
            # Store final reconstruction to connect to output
            final_reconstruction = phi_output  # Initialize with phi output
            
            # Inner optimization loop with actual gradient steps
            for iteration in range(self.inner_iterations):
                with tf.GradientTape() as tape:
                    # Forward pass through inner encoder
                    inner_transformed = self.inner_encoder(phi_output, training=training)
                    
                    # Decode back to original space
                    reconstructed = self.g_dense(inner_transformed) + self.g_bias
                    
                    # Apply layer normalization if enabled
                    if self.decoder_ln:
                        reconstructed = self.layer_norm(reconstructed, training=training)
                    
                    # Compute reconstruction loss
                    inner_loss = tf.reduce_mean(tf.square(reconstructed - target))
                    inner_losses.append(inner_loss)
                
                # Compute gradients with respect to inner encoder parameters
                gradients = tape.gradient(inner_loss, self.inner_encoder.trainable_variables)
                
                # Apply gradient updates (SGD step)
                for var, grad in zip(self.inner_encoder.trainable_variables, gradients):
                    if grad is not None:
                        var.assign_sub(self.inner_lr * grad)
                
                # Keep the final reconstruction to connect to output
                final_reconstruction = reconstructed
            
            # Generate final output by combining both pathways using updated encoder
            # Psi pathway (main pathway)
            psi_output = self.psi_dense(sequence_input)
            inner_transformed = self.inner_encoder(psi_output, training=training)
            
            # Combine with reconstruction pathway to ensure all components get gradients
            # Use a residual-like connection
            combined_features = inner_transformed + 0.1 * final_reconstruction  # Small weight for reconstruction
            final_output = self.h_dense(combined_features) + self.h_bias
            
        finally:
            # Restore original parameters for next time step
            for original_var, current_var in zip(original_weights, self.inner_encoder.trainable_variables):
                current_var.assign(original_var)
        
        return final_output, inner_losses


class TTT_RNN(Model):
    """
    TTT-based RNN model that replaces GRU with TTT cells for phoneme prediction.
    """
    
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=2,
                 conv_kwargs=None,
                 stack_kwargs=None,
                 ttt_config=None,
                 use_enhanced_ttt=False,
                 **kwargs):
        super(TTT_RNN, self).__init__(**kwargs)
        
        self.units = units
        self.weightReg = weightReg
        self.actReg = actReg
        self.subsampleFactor = subsampleFactor
        self.nClasses = nClasses
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.nLayers = nLayers
        self.conv_kwargs = conv_kwargs
        self.stack_kwargs = stack_kwargs
        self.ttt_config = ttt_config or {}
        self.use_enhanced_ttt = use_enhanced_ttt
        
        # Set up regularizers
        self.weight_regularizer = tf.keras.regularizers.L2(weightReg)
        self.kernel_init = tf.keras.initializers.glorot_uniform()
        
        # Initialize TTT-specific configurations
        self.inner_encoder_type = self.ttt_config.get('inner_encoder', 'mlp_2')
        self.inner_iterations = self.ttt_config.get('inner_iterations', 1)
        self.inner_lr = self.ttt_config.get('inner_lr', [0.01])
        self.use_sgd = self.ttt_config.get('use_sgd', True)
        self.decoder_ln = self.ttt_config.get('decoder_ln', True)
        self.sequence_length = self.ttt_config.get('sequence_length', 32)
        
        # Build model components
        self._build_components()
        
    def _build_components(self):
        """Build the model components."""
        # Initial state variables
        if self.bidirectional:
            self.initStates = [
                tf.Variable(initial_value=self.kernel_init(shape=(1, self.units))),
                tf.Variable(initial_value=self.kernel_init(shape=(1, self.units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=self.kernel_init(shape=(1, self.units)))
        
        # Optional convolution layer
        self.conv1 = None
        if self.conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                **self.conv_kwargs,
                padding='same',
                activation='relu',
                kernel_regularizer=self.weight_regularizer,
                use_bias=False
            )
        
        # TTT-RNN layers
        self.ttt_layers = []
        for layer_idx in range(self.nLayers):
            if self.use_enhanced_ttt:
                ttt_cell = EnhancedTTTRNNCell(
                    units=self.units,
                    inner_encoder_type=self.inner_encoder_type,
                    inner_iterations=self.inner_iterations,
                    inner_lr=self.inner_lr,
                    use_sgd=self.use_sgd,
                    decoder_ln=self.decoder_ln,
                    sequence_length=self.sequence_length,
                    name=f"enhanced_ttt_cell_{layer_idx}"
                )
            else:
                ttt_cell = TTTRNNCell(
                    units=self.units,
                    inner_encoder_type=self.inner_encoder_type,
                    inner_iterations=self.inner_iterations,
                    inner_lr=self.inner_lr[0] if isinstance(self.inner_lr, list) else self.inner_lr,
                    use_sgd=self.use_sgd,
                    decoder_ln=self.decoder_ln,
                    name=f"ttt_cell_{layer_idx}"
                )
            
            # Wrap cell in RNN layer
            ttt_rnn = tf.keras.layers.RNN(
                ttt_cell,
                return_sequences=True,
                return_state=True,
                name=f"ttt_rnn_{layer_idx}"
            )
            
            self.ttt_layers.append(ttt_rnn)
            
        # Handle bidirectional case
        if self.bidirectional:
            self.ttt_layers = [
                tf.keras.layers.Bidirectional(layer, name=f"bidirectional_{i}")
                for i, layer in enumerate(self.ttt_layers)
            ]
        
        # Output dense layer
        self.dense = tf.keras.layers.Dense(
            self.nClasses,
            kernel_regularizer=self.weight_regularizer,
            name="output_dense"
        )
        
        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        else:
            self.dropout_layer = None
    
    def call(self, x, states=None, training=False, returnState=False):
        """
        Forward pass through TTT-RNN model.
        
        Args:
            x: Input tensor [batch_size, time_steps, features]
            states: Initial states for RNN layers
            training: Whether in training mode
            returnState: Whether to return final states
            
        Returns:
            output: Model predictions [batch_size, time_steps, nClasses]
            states: Final RNN states (if returnState=True)
        """
        batch_size = tf.shape(x)[0]
        
        # Handle stacked inputs if specified
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(
                x[:, None, :, :],
                sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                strides=[1, 1, self.stack_kwargs['strides'], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            x = tf.squeeze(x, axis=1)
        
        # Apply convolution if specified
        if self.conv1 is not None:
            x = self.conv1(x, training=training)
        
        # Initialize states if not provided
        if states is None:
            states = []
            for i, ttt_layer in enumerate(self.ttt_layers):
                if self.bidirectional:
                    # For bidirectional, we need to handle forward and backward states
                    if self.use_enhanced_ttt:
                        # Enhanced TTT needs [hidden, sequence_buffer] for each direction
                        forward_states = [
                            tf.tile(self.initStates[0], [batch_size, 1]),
                            tf.zeros((batch_size, self.sequence_length, self.units), dtype=x.dtype)
                        ]
                        backward_states = [
                            tf.tile(self.initStates[1], [batch_size, 1]),
                            tf.zeros((batch_size, self.sequence_length, self.units), dtype=x.dtype)
                        ]
                        states.append([forward_states, backward_states])
                    else:
                        # Basic TTT needs just hidden state for each direction
                        states.append([tf.tile(s, [batch_size, 1]) for s in self.initStates])
                else:
                    # For unidirectional
                    if self.use_enhanced_ttt:
                        # Enhanced TTT needs [hidden, sequence_buffer]
                        initial_states = [
                            tf.tile(self.initStates, [batch_size, 1]),
                            tf.zeros((batch_size, self.sequence_length, self.units), dtype=x.dtype)
                        ]
                        states.append(initial_states)
                    else:
                        # Basic TTT needs just hidden state
                        states.append(tf.tile(self.initStates, [batch_size, 1]))
                        
                # For subsequent layers, use None (they will be initialized by the layer)
                if i == 0:
                    continue
                else:
                    states.append(None)
        
        # Forward pass through TTT-RNN layers
        new_states = []
        for i, ttt_layer in enumerate(self.ttt_layers):
            if self.bidirectional:
                result = ttt_layer(
                    x, 
                    training=training, 
                    initial_state=states[i]
                )
                if len(result) == 3:  # x, forward_state, backward_state
                    x, forward_state, backward_state = result
                    new_states.append([forward_state, backward_state])
                else:  # Handle other cases
                    x = result[0]
                    new_states.append(result[1:])
            else:
                result = ttt_layer(x, training=training, initial_state=states[i])
                if len(result) == 2:  # x, state
                    x, state = result
                    new_states.append(state)
                else:  # Handle other cases
                    x = result[0]
                    new_states.append(result[1:])
            
            # Apply subsampling after the second-to-last layer
            if i == len(self.ttt_layers) - 2 and self.subsampleFactor > 1:
                x = x[:, ::self.subsampleFactor, :]
        
        # Apply dropout if specified
        if self.dropout_layer is not None:
            x = self.dropout_layer(x, training=training)
        
        # Final dense layer
        x = self.dense(x, training=training)
        
        if returnState:
            return x, new_states
        else:
            return x
    
    def getSubsampledTimeSteps(self, timeSteps):
        """Compute the number of time steps after subsampling."""
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32
            )
        return timeSteps 