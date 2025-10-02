# --- START OF FILE core/leras/layers/DenseNorm.py ---

# from core.leras import nn # REMOVE
import tensorflow as tf # Import TF directly

# Inherit from Keras Layer
class DenseNorm(tf.keras.layers.Layer): # CORRECT Inheritance
    def __init__(self, dense=False, eps=1e-06, dtype=None, **kwargs):
        # Store eps value
        self.eps_value = eps
        # 'dense' argument doesn't seem to be used? Can be removed if truly unused.
        self.dense_arg_unused = dense # Store if needed, but flag as unused for now

        # Pass dtype up to parent, don't set self.dtype here
        super().__init__(dtype=dtype, **kwargs) # CORRECT - pass dtype up

        # Initialize self.eps in build or call, using self.dtype
        self.eps = None

    # build is simple, only needs to ensure the layer dtype is set
    def build(self, input_shape):
        # Create the constant epsilon using the layer's resolved dtype
        layer_dtype = self.dtype or tf.keras.backend.floatx()
        self.eps = tf.constant(self.eps_value, dtype=layer_dtype, name="epsilon")
        super().build(input_shape) # Mark as built

    # forward renamed to call
    def call(self, x):
        # Pixel normalization logic
        # Calculate mean square along the last axis (features)
        mean_sq = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        # Apply normalization: x / sqrt(mean_square + epsilon)
        return x * tf.math.rsqrt(mean_sq + self.eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dense': self.dense_arg_unused, # Include if needed, otherwise remove
            'eps': self.eps_value,
        })
        return config

# --- REMOVE assignment back to nn.DenseNorm --- CORRECT
# nn.DenseNorm = DenseNorm
# --- END OF FILE ---