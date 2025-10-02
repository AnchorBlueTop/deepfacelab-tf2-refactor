import tensorflow as tf
import numpy as np
import math
import sys

class WScaleDense(tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 gain=1.6, # Empirically chosen higher gain to combat variance decay
                 **kwargs):
        """
        Dense layer with WScale (Equalized Learning Rate).

        Weights are initialized with N(0,1) and then scaled at runtime.
        The `kernel_initializer` will be forced to 'random_normal'.
        The `bias_initializer` will be forced to 'zeros'.

        Args:
            units: Positive integer, dimensionality of the output space.
            gain: The scaling factor for wscale, typically sqrt(2.0).
            **kwargs: Standard Dense arguments (activation, use_bias, etc.)
        """
        if 'kernel_initializer' in kwargs:
            print(f"INFO: WScaleDense overrides user-specified kernel_initializer '{kwargs['kernel_initializer']}' with 'random_normal'.")
        if 'bias_initializer' in kwargs and kwargs.get('use_bias', True):
            print(f"INFO: WScaleDense overrides user-specified bias_initializer '{kwargs['bias_initializer']}' with 'zeros'.")

        super().__init__(
            units=units,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer=tf.keras.initializers.Zeros() if kwargs.get('use_bias', True) else None,
            **kwargs
        )
        self.gain = gain
        self.runtime_scale = None # Will be computed in build

    def build(self, input_shape):
        super().build(input_shape) # Let the parent build self.kernel and self.bias

        # Calculate fan_in for the kernel
        # kernel shape for Dense is (input_dim, units)
        fan_in = self.kernel.shape[0] # input_dim
        
        # Calculate runtime scale factor
        self.runtime_scale = self.gain * tf.math.rsqrt(tf.cast(fan_in, tf.float32))

        # print(f"DEBUG WScaleDense ({self.name}): Built. Fan_in: {fan_in}, Runtime_scale: {self.runtime_scale.numpy()}")

    def call(self, inputs):
        if self.runtime_scale is None:
            if not self.built and hasattr(inputs, 'shape'):
                self.build(inputs.shape)
            if self.runtime_scale is None:
                raise ValueError(f"WScaleDense layer '{self.name}' has not been built properly or runtime_scale is not set.")

        # # ---- WScaleDense Call Debug ---- (COMMENTED OUT)
        # tf.print(f"DEBUG WScaleDense ({self.name}) CALL:")
        # tf.print(f"    inputs min/max/mean/std:", tf.reduce_min(inputs), tf.reduce_max(inputs), tf.reduce_mean(inputs), tf.math.reduce_std(inputs), output_stream=sys.stdout)
        # tf.print(f"    self.kernel min/max/mean/std:", tf.reduce_min(self.kernel), tf.reduce_max(self.kernel), tf.reduce_mean(self.kernel), tf.math.reduce_std(self.kernel), output_stream=sys.stdout)
        # tf.print(f"    self.runtime_scale:", self.runtime_scale, output_stream=sys.stdout)
        # # --------------------------------

        scaled_kernel = self.kernel * self.runtime_scale

        # # ---- WScaleDense Scaled Kernel Debug ---- (COMMENTED OUT)
        # tf.print(f"    scaled_kernel min/max/mean/std:", tf.reduce_min(scaled_kernel), tf.reduce_max(scaled_kernel), tf.reduce_mean(scaled_kernel), tf.math.reduce_std(scaled_kernel), output_stream=sys.stdout)
        # # -------------------------------------

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            outputs = tf.matmul(inputs, scaled_kernel)
        else: 
            outputs = tf.tensordot(inputs, scaled_kernel, [[rank - 1], [0]])

        if self.use_bias:
            # # ---- WScaleDense Bias Debug ---- (COMMENTED OUT)
            # tf.print(f"    self.bias values (if use_bias):", self.bias, output_stream=sys.stdout, summarize=-1)
            # # ------------------------------
            outputs = tf.nn.bias_add(outputs, self.bias)
        
        # # ---- WScaleDense Pre-Activation Output Debug ---- (COMMENTED OUT)
        # tf.print(f"    outputs (pre-activation) min/max/mean/std:", tf.reduce_min(outputs), tf.reduce_max(outputs), tf.reduce_mean(outputs), tf.math.reduce_std(outputs), output_stream=sys.stdout)
        # # -----------------------------------------------
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'gain': self.gain})
        return config