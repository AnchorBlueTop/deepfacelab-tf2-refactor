# --- START OF FILE core/leras/layers/DepthwiseConv2D.py --- (Refactored)

import numpy as np
import tensorflow as tf

# Inherit from Keras Layer
class DepthwiseConv2D(tf.keras.layers.Layer):
    """
    Depthwise Separable Convolution 2D Layer compatible with Keras API.
    Supports custom weight scaling (use_wscale).
    """
    def __init__(self, kernel_size, strides=1, padding='SAME', depth_multiplier=1, dilations=1, use_bias=True, 
                 use_wscale=False, kernel_initializer=None, bias_initializer=None, 
                 trainable=True, dtype=None, name=None, use_fp16 = False, **kwargs ):
        # Note: in_ch removed from signature, inferred in build
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs) # Pass dtype up

        if not isinstance(strides, int): raise ValueError ("strides must be an int type")
        if not isinstance(dilations, int): raise ValueError ("dilations must be an int type")

        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.depth_multiplier = int(depth_multiplier)
        self.dilations = int(dilations)
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer_arg = kernel_initializer
        self.bias_initializer_arg = bias_initializer
        self.use_fp16 = use_fp16

        # Store Keras standard padding string
        if isinstance(padding, str):
            self.padding = padding.upper()
            if self.padding not in ('SAME', 'VALID'):
                print(f"Warning: Non-standard padding '{padding}'. Interpreting as 'SAME'.")
                self.padding = 'SAME'
        elif isinstance(padding, int):
            print(f"Warning: Integer padding '{padding}'. Interpreting as 'SAME'.")
            self.padding = 'SAME'
        else:
             raise ValueError("Padding must be 'SAME', 'VALID', or an integer (interpreted as 'SAME').")


    def build(self, input_shape):
        """Create weights based on input shape."""
        data_format = tf.keras.backend.image_data_format()
        channel_axis = -1 if data_format == 'channels_last' else 1
        self.in_ch = input_shape[channel_axis]
        if self.in_ch is None: raise ValueError('Input channel dimension must be defined.')
        self.in_ch = int(self.in_ch) # Ensure it's an integer

        layer_dtype = self.dtype or tf.keras.backend.floatx()

        # Kernel shape: [H, W, In_C, Depth_Multiplier]
        kernel_shape = (self.kernel_size, self.kernel_size, self.in_ch, self.depth_multiplier)
        kernel_initializer = self.kernel_initializer_arg

        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2.0)
            # Fan in for depthwise is kernel_size * kernel_size * in_ch (input channels, not output)
            fan_in = float(self.kernel_size * self.kernel_size * self.in_ch)
            he_std = gain / np.sqrt(fan_in) if fan_in > 0 else 1.0
            self.wscale = tf.constant(he_std, dtype=layer_dtype)
            if kernel_initializer is None: kernel_initializer = tf.keras.initializers.RandomNormal(0, 1.0)
        else:
            self.wscale = None
            if kernel_initializer is None: kernel_initializer = tf.keras.initializers.GlorotUniform()

        # Use add_weight for the depthwise kernel
        self.depthwise_kernel = self.add_weight(
            name='depthwise_kernel', # Keras convention
            shape=kernel_shape,
            initializer=kernel_initializer,
            trainable=self.trainable,
            dtype=layer_dtype
        )

        if self.use_bias:
            bias_initializer = self.bias_initializer_arg or tf.keras.initializers.Zeros()
            # Bias shape for depthwise is In_C * Depth_Multiplier
            self.bias = self.add_weight(
                name='bias',
                shape=(self.in_ch * self.depth_multiplier,),
                initializer=bias_initializer,
                trainable=self.trainable,
                dtype=layer_dtype
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        """Apply depthwise convolution."""
        data_format = tf.keras.backend.image_data_format()
        data_format_string = "NHWC" if data_format == 'channels_last' else "NCHW"

        effective_kernel = self.depthwise_kernel
        if self.use_wscale and self.wscale is not None:
             wscale_casted = tf.cast(self.wscale, self.depthwise_kernel.dtype)
             effective_kernel = self.depthwise_kernel * wscale_casted

        # Prepare strides list based on data format
        if data_format == 'channels_last': # NHWC
            strides_list = [1, self.strides, self.strides, 1]
            dilations_list = [self.dilations, self.dilations] # TF expects 2D for depthwise dilations
        else: # NCHW
            strides_list = [1, 1, self.strides, self.strides]
            dilations_list = [self.dilations, self.dilations] # TF expects 2D

        outputs = tf.nn.depthwise_conv2d(
                inputs,
                effective_kernel,
                strides=strides_list,
                padding=self.padding, # Use 'SAME' or 'VALID' string
                data_format=data_format_string,
                dilations=dilations_list # Pass dilations
            )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=data_format_string)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            # 'in_ch': self.in_ch, # Not typically stored, inferred in build
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'depth_multiplier': self.depth_multiplier,
            'dilations': self.dilations,
            'use_bias': self.use_bias,
            'use_wscale': self.use_wscale,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer_arg),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer_arg),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['kernel_initializer'] = tf.keras.initializers.deserialize(config.get('kernel_initializer'))
        config['bias_initializer'] = tf.keras.initializers.deserialize(config.get('bias_initializer'))
        return cls(**config)

    def __str__(self):
        return f"{self.__class__.__name__} : in_ch:{self.in_ch if self.built else '?'} mult:{self.depth_multiplier} kernel:{self.kernel_size} stride:{self.strides}"

# --- DO NOT ASSIGN back to nn.DepthwiseConv2D ---
# --- END OF FILE ---