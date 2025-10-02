# --- START OF FILE core/leras/layers/Dense.py ---

import numpy as np
from core.leras import nn
# Ensure the main tensorflow module is imported, not compat.v1
import tensorflow as tf

# Inherit from tf.keras.layers.Layer
class Dense(tf.keras.layers.Layer):
    """
    Fully connected layer compatible with Keras API.

    Supports custom weight scaling (use_wscale) and Maxout activation.
    """
    def __init__(self, in_ch, units, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, 
                 bias_initializer=None, trainable=True, dtype=None, name=None, use_fp16=False, **kwargs ): 
                 # Added use_fp16 here temporarily
        """
        Args:
            in_ch: (Deprecated, kept for signature compatibility) Input channels, inferred in build().
            units: Integer, the number of output units (equivalent to original out_ch).
            use_bias: Boolean, whether the layer uses a bias vector.
            use_wscale: Boolean, enables weight scale (equalized learning rate).
                        If kernel_initializer is None, it will be forced to random_normal.
            maxout_ch: Integer, number of channels for maxout operation (0 or 1 to disable).
                       See: https://link.springer.com/article/10.1186/s40537-019-0233-0
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            trainable: Boolean, if the layer's variables should be trainable.
            dtype: Optional dtype of the layer's weights.
            name: Optional name for the layer.
            **kwargs: Additional keyword arguments passed to the parent Layer.
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self.units = int(units) # Renamed from out_ch for Keras convention
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = int(maxout_ch)
        self.kernel_initializer_arg = kernel_initializer # Store arg for build
        self.bias_initializer_arg = bias_initializer   # Store arg for build
        self._original_in_ch = in_ch # Store original if needed, e.g., for wscale before build
        self.use_fp16 = use_fp16

        if self.maxout_ch < 0:
            raise ValueError("maxout_ch must be >= 0")
        if self.maxout_ch == 1: # Maxout with 1 channel is just a normal dense layer
            self.maxout_ch = 0


    def build(self, input_shape):
        # Infer input dimension from the last axis of input_shape
        if not isinstance(input_shape, tf.TensorShape):
             input_shape = tf.TensorShape(input_shape)
        last_dim = input_shape[-1]
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')
        self.in_ch = int(last_dim) # Inferred input channels

        # --- Kernel (Weight) Initialization ---
        if self.maxout_ch > 1:
            kernel_shape = (self.in_ch, self.units * self.maxout_ch)
        else:
            kernel_shape = (self.in_ch, self.units)

        kernel_initializer = self.kernel_initializer_arg # Use the stored arg

        if self.use_wscale:
            gain = 1.0
            fan_in = float(self.in_ch) # Fan in is just the input dimension for Dense
            he_std = gain / np.sqrt(fan_in)
            self.wscale = tf.constant(he_std, dtype=self.dtype) # Store wscale factor

            # If using wscale and no initializer provided, use random normal
            if kernel_initializer is None:
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        else:
            self.wscale = None
            # If not using wscale and no initializer provided, use default Keras (GlorotUniform)
            if kernel_initializer is None:
                 kernel_initializer = tf.keras.initializers.GlorotUniform()

        # Use add_weight() to create the kernel variable
        self.kernel = self.add_weight( # Keras convention is 'kernel'
            name='kernel',
            shape=kernel_shape,
            initializer=kernel_initializer,
            trainable=self.trainable,
            dtype=self.dtype)

        # --- Bias Initialization ---
        if self.use_bias:
            bias_initializer = self.bias_initializer_arg # Use the stored arg
            if bias_initializer is None:
                bias_initializer = tf.keras.initializers.Zeros()

            # Use add_weight() to create the bias variable
            # Bias shape is always (units,) regardless of maxout
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=bias_initializer,
                trainable=self.trainable,
                dtype=self.dtype)
        else:
            self.bias = None

        # Ensure Keras knows the layer is built
        super().build(input_shape)

    # Replaces 'forward' method
    def call(self, inputs):
        # Apply wscale if enabled
        effective_kernel = self.kernel
        if self.use_wscale and self.wscale is not None:
            effective_kernel = self.kernel * self.wscale

        # Perform matrix multiplication
        # tf.matmul automatically handles broadcasting for batch dimension
        outputs = tf.matmul(inputs, effective_kernel)

        # Apply Maxout if enabled
        if self.maxout_ch > 1:
            # Reshape to (batch_size, units, maxout_ch) and take max along the last axis
            outputs = tf.reshape (outputs, (-1, self.units, self.maxout_ch) )
            outputs = tf.reduce_max(outputs, axis=-1) # Result shape (batch_size, units)

        # Add bias if enabled
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias) # bias_add handles broadcasting

        return outputs

    # Optional: Implement get_config for Keras saving/loading compatibility
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_ch': self._original_in_ch, # Store original arg if needed
            'units': self.units,
            'use_bias': self.use_bias,
            'use_wscale': self.use_wscale,
            'maxout_ch': self.maxout_ch,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer_arg),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer_arg),
            # 'trainable' and 'dtype' are handled by the base Layer get_config
        })
        return config

    def __str__(self):
        return f"{self.__class__.__name__} (in: {self.in_ch if self.built else '?'}, units: {self.units}, maxout: {self.maxout_ch})"


# Assign the refactored class back to nn.Dense for compatibility
# nn.Dense = Dense
# --- END OF FILE core/leras/layers/Dense.py ---