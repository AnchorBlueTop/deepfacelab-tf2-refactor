# --- START OF FILE core/leras/layers/InstanceNorm2D.py --- (Refactored)

# from core.leras import nn # REMOVE
import tensorflow as tf

# Inherit from Keras Layer
class InstanceNorm2D(tf.keras.layers.Layer):
    """
    Instance Normalization Layer (Keras implementation).
    Normalizes along spatial dimensions (H, W) for each channel and sample.
    """
    def __init__(self, epsilon=1e-5, name=None, dtype=None, **kwargs):
        """
        Args:
            epsilon: Small float added to variance to avoid dividing by zero.
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        # Note: in_ch inferred in build, removed from init signature
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.epsilon = epsilon
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """Create learnable scale and shift parameters."""
        if not self.built:
            data_format = tf.keras.backend.image_data_format()
            channel_axis = -1 if data_format == 'channels_last' else 1
            shape = tf.TensorShape(input_shape)
            if shape[channel_axis] is None:
                raise ValueError("The channel dimension of the inputs must be defined.")
            in_ch = int(shape[channel_axis])

            layer_dtype = self.dtype or tf.keras.backend.floatx()

            # Add learnable gamma (scale) and beta (shift) variables
            self.gamma = self.add_weight(
                name="gamma",
                shape=(in_ch,),
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
                dtype=layer_dtype
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(in_ch,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=layer_dtype
            )
        super().build(input_shape)

    def call(self, inputs):
        """Apply Instance Normalization."""
        data_format = tf.keras.backend.image_data_format()
        spatial_axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        param_shape = [1, 1, 1, -1] if data_format == 'channels_last' else [1, -1, 1, 1] # For broadcasting gamma/beta

        # Calculate mean and variance over spatial dimensions for each channel/sample
        mean = tf.reduce_mean(inputs, axis=spatial_axes, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=spatial_axes, keepdims=True)

        # Normalize: (inputs - mean) / sqrt(variance + epsilon)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv

        # Apply scale and shift
        scale = tf.reshape(self.gamma, param_shape)
        shift = tf.reshape(self.beta, param_shape)

        return scale * normalized + shift

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

# --- REMOVE assignment back ---
# nn.InstanceNorm2D = InstanceNorm2D
# --- END OF FILE ---