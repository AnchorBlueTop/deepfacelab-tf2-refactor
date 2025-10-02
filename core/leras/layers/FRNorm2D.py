# --- START OF FILE core/leras/layers/FRNorm2D.py --- (Refactored)

# from core.leras import nn # REMOVE
import tensorflow as tf

# Inherit from Keras Layer
class FRNorm2D(tf.keras.layers.Layer):
    """
    Filter Response Normalization Layer (Keras implementation).
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, eps=1e-6, dtype=None, name=None, **kwargs):
        """
        Args:
            eps: Epsilon value for numerical stability.
            dtype: Data type for weights. Defaults to Keras backend floatx.
            name: Layer name.
            **kwargs: Additional Layer arguments.
        """
        # Note: in_ch is inferred in build(), removed from __init__ signature
        super().__init__(name=name, dtype=dtype, **kwargs)
        # Store eps as a float, will be cast to layer dtype if needed
        self.eps_value = eps
        # Initialize weight/bias/eps attributes to None, created in build()
        self.gamma = None # Keras convention uses gamma for learnable scale
        self.beta = None  # Keras convention uses beta for learnable shift/bias
        self.learned_eps = None # Learnable epsilon

    def build(self, input_shape):
        """Create weights based on input channels."""
        if not self.built: # Ensure build runs only once
            # Get data format and determine channel axis
            data_format = tf.keras.backend.image_data_format()
            channel_axis = -1 if data_format == 'channels_last' else 1
            # Infer input channels
            if input_shape[channel_axis] is None:
                raise ValueError("The channel dimension of the inputs must be defined.")
            in_ch = int(input_shape[channel_axis])

            # Use add_weight to create learnable parameters
            self.gamma = self.add_weight(
                name="gamma", # Equivalent to original 'weight'
                shape=(in_ch,),
                initializer=tf.keras.initializers.Ones(), # Initialize scale to 1
                trainable=True,
                dtype=self.dtype # Use layer's dtype
            )
            self.beta = self.add_weight(
                name="beta", # Equivalent to original 'bias'
                shape=(in_ch,),
                initializer=tf.keras.initializers.Zeros(), # Initialize shift to 0
                trainable=True,
                dtype=self.dtype
            )
            # Use add_weight for the learnable epsilon
            # Initialize near the provided eps value, ensure it stays positive
            self.learned_eps = self.add_weight(
                name="epsilon",
                shape=(1,), # Single value epsilon
                initializer=tf.keras.initializers.Constant(self.eps_value),
                # Constraint to keep epsilon positive during training
                constraint=tf.keras.constraints.NonNeg(),
                trainable=True, # Make epsilon learnable as per paper? Check if desired.
                dtype=self.dtype
            )
        # Mark as built
        super().build(input_shape)

    # Rename forward to call
    def call(self, x):
        """Apply Filter Response Normalization."""
        # Get data format and spatial axes dynamically
        data_format = tf.keras.backend.image_data_format()
        spatial_axes = [1, 2] if data_format == 'channels_last' else [2, 3]

        # Calculate mean squared over spatial dimensions (nu^2 in paper)
        nu2 = tf.reduce_mean(tf.square(x), axis=spatial_axes, keepdims=True)

        # Normalize input: x / sqrt(nu^2 + |epsilon|)
        # Use the learnable epsilon, ensure positivity with abs()
        x_normalized = x * tf.math.rsqrt(nu2 + tf.abs(self.learned_eps))

        # Reshape scale (gamma) and shift (beta) for broadcasting
        param_shape = [1, 1, 1, -1] if data_format == 'channels_last' else [1, -1, 1, 1]
        gamma_reshaped = tf.reshape(self.gamma, param_shape)
        beta_reshaped = tf.reshape(self.beta, param_shape)

        # Apply scale and shift: gamma * x_normalized + beta
        return gamma_reshaped * x_normalized + beta_reshaped

    def get_config(self):
        """Serialize layer configuration."""
        config = super().get_config()
        config.update({
            'eps': self.eps_value, # Store initial epsilon
        })
        return config

# --- REMOVE assignment back to nn.FRNorm2D ---
# nn.FRNorm2D = FRNorm2D
# --- END OF FILE ---