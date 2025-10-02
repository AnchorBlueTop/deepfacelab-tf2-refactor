# --- START OF FILE core/leras/layers/BatchNorm2D.py --- (Refactored)

# from core.leras import nn # REMOVE
import tensorflow as tf

# Inherit from Keras Layer
class BatchNorm2D(tf.keras.layers.Layer):
    """
    Batch Normalization 2D Layer (Keras implementation mimicking inference mode).

    NOTE: This implementation primarily replicates the *inference* behavior
    of the original using loaded running stats. For standard trainable
    batch normalization, use tf.keras.layers.BatchNormalization.
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, name=None, dtype=None, **kwargs):
        """
        Args:
            dim: Integer, the number of channels/features.
            eps: Epsilon for numerical stability.
            momentum: Momentum for moving averages (unused in this inference-only setup).
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        super().__init__(name=name, dtype=dtype, trainable=False, **kwargs) # Set trainable=False
        self.dim = dim
        self.eps = eps
        self.momentum = momentum # Stored but not used by this inference impl.
        # Weights/stats are created in build()
        self.gamma = None # Keras convention for scale
        self.beta = None  # Keras convention for shift
        self.moving_mean = None # Keras convention
        self.moving_variance = None # Keras convention

    def build(self, input_shape):
        """Creates non-trainable weights for inference."""
        if not self.built:
            layer_dtype = self.dtype or tf.keras.backend.floatx()

            # Create variables using add_weight, matching Keras conventions
            self.gamma = self.add_weight(
                name="gamma", # Equivalent to original 'weight'
                shape=(self.dim,),
                initializer=tf.keras.initializers.Ones(),
                trainable=False, # Non-trainable for inference mimic
                dtype=layer_dtype
            )
            self.beta = self.add_weight(
                name="beta", # Equivalent to original 'bias'
                shape=(self.dim,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
                dtype=layer_dtype
            )
            self.moving_mean = self.add_weight(
                name="moving_mean", # Equivalent to original 'running_mean'
                shape=(self.dim,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False, # Non-trainable stats
                dtype=layer_dtype
            )
            self.moving_variance = self.add_weight(
                name="moving_variance", # Equivalent to original 'running_var'
                shape=(self.dim,),
                # Initialize variance to 1 for inference (safer than 0)
                initializer=tf.keras.initializers.Ones(),
                trainable=False,
                dtype=layer_dtype
            )
        super().build(input_shape)

    # Renamed forward to call
    def call(self, x, training=None): # Add training argument (ignored here)
        """Applies Batch Normalization using stored statistics (inference mode)."""
        # Get data format and reshape parameters
        data_format = tf.keras.backend.image_data_format()
        param_shape = [1, 1, 1, self.dim] if data_format == 'channels_last' else [1, self.dim, 1, 1]

        gamma_reshaped = tf.reshape(self.gamma, param_shape)
        beta_reshaped = tf.reshape(self.beta, param_shape)
        mean_reshaped = tf.reshape(self.moving_mean, param_shape)
        variance_reshaped = tf.reshape(self.moving_variance, param_shape)

        # Apply normalization using stored mean and variance
        # y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        outputs = tf.nn.batch_normalization(
            x,
            mean=mean_reshaped,
            variance=variance_reshaped,
            offset=beta_reshaped,
            scale=gamma_reshaped,
            variance_epsilon=self.eps
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'eps': self.eps,
            'momentum': self.momentum,
        })
        return config

# --- DO NOT ASSIGN back to nn.BatchNorm2D here ---
# --- END OF FILE ---