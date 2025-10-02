import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.python.keras.utils import conv_utils # For data_format conversion

class WScaleConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 gain=1.6, # Empirically chosen higher gain to combat variance decay
                 **kwargs):
        """
        Conv2D layer with WScale (Equalized Learning Rate).

        Weights are initialized with N(0,1) and then scaled at runtime.
        The `kernel_initializer` will be forced to 'random_normal'.
        The `bias_initializer` will be forced to 'zeros'.

        Args:
            filters: Integer, the dimensionality of the output space.
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                         height and width of the 2D convolution window.
            gain: The scaling factor for wscale, typically sqrt(2.0).
            **kwargs: Standard Conv2D arguments (strides, padding, activation, etc.)
        """
        if 'kernel_initializer' in kwargs:
            print(f"INFO: WScaleConv2D overrides user-specified kernel_initializer '{kwargs['kernel_initializer']}' with 'random_normal'.")
        if 'bias_initializer' in kwargs and kwargs.get('use_bias', True):
            print(f"INFO: WScaleConv2D overrides user-specified bias_initializer '{kwargs['bias_initializer']}' with 'zeros'.")

        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer=tf.keras.initializers.Zeros() if kwargs.get('use_bias', True) else None,
            **kwargs
        )
        self.gain = gain
        self.runtime_scale = None # Will be computed in build

    def build(self, input_shape):
        super().build(input_shape) # Let the parent build self.kernel and self.bias

        # Calculate fan_in for the kernel
        # kernel shape is (kh, kw, input_channels, output_channels)
        fan_in = np.prod(self.kernel.shape[:-1]) # kh * kw * input_channels
        
        # Calculate runtime scale factor
        self.runtime_scale = self.gain * tf.math.rsqrt(tf.cast(fan_in, tf.float32))
        
        # Ensure kernel is trainable (it should be by default)
        # print(f"DEBUG WScaleConv2D ({self.name}): Kernel trainable: {self.kernel.trainable}")
        # print(f"DEBUG WScaleConv2D ({self.name}): Built. Fan_in: {fan_in}, Runtime_scale: {self.runtime_scale.numpy()}")


    def call(self, inputs):
        if self.runtime_scale is None:
            if not self.built and hasattr(inputs, 'shape'):
                self.build(inputs.shape)
            if self.runtime_scale is None: 
                raise ValueError(f"WScaleConv2D layer '{self.name}' has not been built properly or runtime_scale is not set.")

        # # ---- WScaleConv2D Call Debug ---- (COMMENTED OUT)
        # tf.print(f"DEBUG WScaleConv2D ({self.name}) CALL:")
        # tf.print(f"    inputs min/max/mean/std:", tf.reduce_min(inputs), tf.reduce_max(inputs), tf.reduce_mean(inputs), tf.math.reduce_std(inputs), output_stream=sys.stdout)
        # tf.print(f"    self.kernel min/max/mean/std:", tf.reduce_min(self.kernel), tf.reduce_max(self.kernel), tf.reduce_mean(self.kernel), tf.math.reduce_std(self.kernel), output_stream=sys.stdout)
        # tf.print(f"    self.runtime_scale:", self.runtime_scale, output_stream=sys.stdout)
        # # ---------------------------------

        scaled_kernel = self.kernel * self.runtime_scale
        
        # # ---- WScaleConv2D Scaled Kernel Debug ---- (COMMENTED OUT)
        # tf.print(f"    scaled_kernel min/max/mean/std:", tf.reduce_min(scaled_kernel), tf.reduce_max(scaled_kernel), tf.reduce_mean(scaled_kernel), tf.math.reduce_std(scaled_kernel), output_stream=sys.stdout)
        # # ----------------------------------------
        
        tf_data_format = conv_utils.convert_data_format(self.data_format, ndim=4)

        outputs = tf.nn.conv2d(
            inputs,
            scaled_kernel, 
            strides=list(self.strides), 
            padding=self.padding.upper(),
            data_format=tf_data_format,
            dilations=list(self.dilation_rate)
        )

        if self.use_bias:
            # # ---- WScaleConv2D Bias Debug ---- (COMMENTED OUT)
            # tf.print(f"    self.bias values (if use_bias):", self.bias, output_stream=sys.stdout, summarize=-1) # Summarize -1 to print all bias values
            # # ---------------------------------
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=tf_data_format)
        
        # # ---- WScaleConv2D Pre-Activation Output Debug ---- (COMMENTED OUT)
        # tf.print(f"    outputs (pre-activation) min/max/mean/std:", tf.reduce_min(outputs), tf.reduce_max(outputs), tf.reduce_mean(outputs), tf.math.reduce_std(outputs), output_stream=sys.stdout)
        # # ------------------------------------------------

        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'gain': self.gain})
        # kernel_initializer and bias_initializer are handled by super,
        # but they are fixed in our __init__.
        return config