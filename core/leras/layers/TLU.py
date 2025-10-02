# --- START OF FILE core/leras/layers/TLU.py --- (Refactored)

import tensorflow as tf

class TLU(tf.keras.layers.Layer):
    """
    Thresholded Linear Unit (TLU) Layer (Keras implementation).
    From: Filter Response Normalization Layer: Eliminating Batch Dependence...
    https://arxiv.org/pdf/1911.09737.pdf
    Output = max(x, tau), where tau is a learnable threshold per channel.
    """
    def __init__(self, name=None, dtype=None, **kwargs):
        """
        Args:
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        # Note: in_ch is inferred in build()
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.tau = None # Learnable threshold created in build

    def build(self, input_shape):
        """Create the learnable threshold 'tau'."""
        if not self.built:
            data_format = tf.keras.backend.image_data_format()
            channel_axis = -1 if data_format == 'channels_last' else 1
            if input_shape[channel_axis] is None:
                 raise ValueError("The channel dimension of the inputs must be defined.")
            in_ch = int(input_shape[channel_axis])

            layer_dtype = self.dtype or tf.keras.backend.floatx()

            # Create the learnable threshold tau
            self.tau = self.add_weight(
                name="tau",
                shape=(in_ch,),
                initializer=tf.keras.initializers.Zeros(), # Initialize threshold to 0
                trainable=True,
                dtype=layer_dtype
            )
        super().build(input_shape)

    def call(self, x):
        """Apply Thresholded Linear Unit activation."""
        # Reshape tau for broadcasting
        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_last': # NHWC
            # Target shape (1, 1, 1, C)
            param_shape = tf.concat( ([1]*(x.shape.rank-1), [tf.shape(self.tau)[0]]), axis=0 )
        else: # NCHW
            # Target shape (1, C, 1, 1)
            param_shape = tf.concat( ([1], [tf.shape(self.tau)[0]], [1]*(x.shape.rank-2)), axis=0 )

        tau_reshaped = tf.reshape(self.tau, param_shape)

        # Apply max(x, tau)
        return tf.maximum(x, tau_reshaped)

    def get_config(self):
        # No extra config needed beyond base Layer args
        config = super().get_config()
        return config

# --- REMOVE assignment back ---
# nn.TLU = TLU
# --- END OF FILE ---