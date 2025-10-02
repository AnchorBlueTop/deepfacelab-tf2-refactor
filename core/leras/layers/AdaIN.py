# --- START OF FILE core/leras/layers/AdaIN.py --- (Refactored)

import tensorflow as tf
# Removed: from core.leras import nn

# Adaptive Instance Normalization Layer
class AdaIN(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization layer (Keras implementation).
    Applies style (from MLP) to content features (x).
    """
    def __init__(self, mlp_ch, kernel_initializer=None, name=None, dtype=None, **kwargs):
        """
        Args:
            mlp_ch: Integer, the number of channels in the input MLP style vector.
            kernel_initializer: Initializer for the Dense kernel weights.
                                Defaults to HeNormal if None.
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        # Note: in_ch is inferred in build()
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.mlp_ch = mlp_ch
        self.kernel_initializer_arg = kernel_initializer
        # Weights are created in build()
        self.weight_gamma = None
        self.bias_gamma = None
        self.weight_beta = None
        self.bias_beta = None

    def build(self, input_shape):
        """Creates weights based on input shapes."""
        # Input shape is expected to be a list/tuple: [content_shape, mlp_shape]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("AdaIN layer expects a list/tuple of two input shapes: [content_shape, mlp_shape]")

        content_shape = tf.TensorShape(input_shape[0])
        mlp_shape = tf.TensorShape(input_shape[1])

        # Infer content input channels
        data_format = tf.keras.backend.image_data_format()
        channel_axis = -1 if data_format == 'channels_last' else 1
        in_ch = content_shape[channel_axis]
        if in_ch is None:
             raise ValueError("The channel dimension of the content input must be defined.")
        in_ch = int(in_ch)

        # Verify MLP input shape
        if mlp_shape.rank != 2:
             raise ValueError("MLP input must be 2D (batch, mlp_ch)")
        if mlp_shape[-1] != self.mlp_ch:
             raise ValueError(f"MLP input channel dimension ({mlp_shape[-1]}) does not match mlp_ch ({self.mlp_ch})")

        # Get layer dtype
        layer_dtype = self.dtype or tf.keras.backend.floatx()

        # Determine initializers
        kernel_initializer = self.kernel_initializer_arg
        if kernel_initializer is None:
            # Use HeNormal similar to original logic if none provided
            kernel_initializer = tf.keras.initializers.HeNormal()
        bias_initializer = tf.keras.initializers.Zeros()

        # Create weights using add_weight
        self.weight_gamma = self.add_weight(
            name="weight_gamma",
            shape=(self.mlp_ch, in_ch),
            initializer=kernel_initializer,
            dtype=layer_dtype,
            trainable=self.trainable # Inherited from Layer
        )
        self.bias_gamma = self.add_weight(
            name="bias_gamma",
            shape=(in_ch,),
            initializer=bias_initializer,
            dtype=layer_dtype,
            trainable=self.trainable
        )
        self.weight_beta = self.add_weight(
            name="weight_beta",
            shape=(self.mlp_ch, in_ch),
            initializer=kernel_initializer,
            dtype=layer_dtype,
            trainable=self.trainable
        )
        self.bias_beta = self.add_weight(
            name="bias_beta",
            shape=(in_ch,),
            initializer=bias_initializer,
            dtype=layer_dtype,
            trainable=self.trainable
        )
        super().build(input_shape) # Mark layer as built

    # Renamed forward to call
    def call(self, inputs):
        """Applies AdaIN."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("AdaIN layer expects a list/tuple of two inputs: [content_features, mlp_style]")
        x, mlp = inputs
        in_ch = tf.shape(x)[-1] if tf.keras.backend.image_data_format() == 'channels_last' else tf.shape(x)[1]

        # Ensure MLP is float32 for matmul if weights are float32 (or vice versa if needed)
        if mlp.dtype != self.weight_gamma.dtype:
            mlp = tf.cast(mlp, self.weight_gamma.dtype)

        # Calculate scale (gamma) and shift (beta) from MLP style vector
        gamma = tf.matmul(mlp, self.weight_gamma) + self.bias_gamma # Bias added automatically
        beta = tf.matmul(mlp, self.weight_beta) + self.bias_beta   # Bias added automatically

        # Reshape gamma and beta for broadcasting over spatial dimensions
        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_last': # NHWC
            param_shape = tf.stack([-1, 1, 1, in_ch]) # Use dynamic batch size
        else: # NCHW
            param_shape = tf.stack([-1, in_ch, 1, 1])
        gamma = tf.reshape(gamma, param_shape)
        beta = tf.reshape(beta, param_shape)

        # Apply Instance Normalization
        spatial_axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        x_mean = tf.reduce_mean(x, axis=spatial_axes, keepdims=True)
        x_var = tf.math.reduce_variance(x, axis=spatial_axes, keepdims=True) # Use variance
        x_normalized = (x - x_mean) * tf.math.rsqrt(x_var + 1e-5) # Use variance and rsqrt

        # Apply style (scale and shift)
        stylized_x = gamma * x_normalized + beta
        return stylized_x

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_ch': self.mlp_ch,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer_arg),
            # Bias initializer usually defaults to Zeros, may not need serialization
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['kernel_initializer'] = tf.keras.initializers.deserialize(config.get('kernel_initializer'))
        return cls(**config)

# --- DO NOT ASSIGN back to nn.AdaIN here ---
# --- END OF FILE ---