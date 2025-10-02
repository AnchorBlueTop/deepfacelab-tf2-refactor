# --- START OF FILE core/leras/layers/Conv2DTranspose.py --- (Corrected Imports/Access)

import numpy as np
# from core.leras import nn # REMOVE
import tensorflow as tf # Import TF directly

# Inherit directly from Keras base class
class Conv2DTranspose(tf.keras.layers.Layer):
    """
    Convolutional Transpose 2D Layer compatible with Keras API.
    Supports custom weight scaling (use_wscale).
    """
    def __init__(self, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, 
                 use_wscale=False, kernel_initializer=None, bias_initializer=None, 
                 trainable=True, dtype=None, name=None, use_fp16 = False, **kwargs ):
         # Note: Removed in_ch from signature, inferred in build
         super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs) # Pass dtype to parent

         self.out_ch = out_ch
         self.kernel_size = int(kernel_size)
         self.strides = int(strides)
         self.padding = padding.upper() # Use Keras standard strings ('SAME', 'VALID')
         self.use_bias = use_bias
         self.use_wscale = use_wscale
         self.kernel_initializer_arg = kernel_initializer
         self.bias_initializer_arg = bias_initializer
         self.use_fp16 = use_fp16
         # Dtype is handled by the parent Layer using the 'dtype' argument

    def build(self, input_shape):
        # Get data_format from Keras backend
        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_last': # NHWC
            channel_axis = -1
        else: # channels_first (NCHW)
            channel_axis = 1
        self.in_ch = input_shape[channel_axis] # Infer in_ch
        if self.in_ch is None: raise ValueError('Input channel dimension must be defined.')

        # Keras Conv2DTranspose kernel shape is (H, W, Out, In)
        kernel_shape = (self.kernel_size, self.kernel_size, self.out_ch, self.in_ch)
        kernel_initializer = self.kernel_initializer_arg

        # Use self.dtype inherited from the Layer base class
        layer_dtype = self.dtype or tf.keras.backend.floatx()

        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2.0)
            fan_in = float(self.kernel_size * self.kernel_size * self.in_ch)
            # Ensure division by zero doesn't happen if fan_in is 0 (though unlikely for conv)
            he_std = gain / np.sqrt(fan_in) if fan_in > 0 else 1.0
            self.wscale = tf.constant(he_std, dtype=layer_dtype ) # Use layer's dtype
            if kernel_initializer is None:
                kernel_initializer = tf.keras.initializers.RandomNormal(0, 1.0)
        else:
            self.wscale = None
            if kernel_initializer is None:
                 kernel_initializer = tf.keras.initializers.GlorotUniform()

        # Use add_weight() to create the kernel variable
        self.kernel = self.add_weight(name='kernel', shape=kernel_shape, initializer=kernel_initializer, trainable=self.trainable, dtype=layer_dtype) # Use layer's dtype
        if self.use_bias:
            bias_initializer = self.bias_initializer_arg or tf.keras.initializers.Zeros()
            self.bias = self.add_weight(name='bias', shape=(self.out_ch,), initializer=bias_initializer, trainable=self.trainable, dtype=layer_dtype ) # Use layer's dtype
        else:
             self.bias = None
        super().build(input_shape)


    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]

        # Get data_format from Keras backend dynamically inside call
        data_format = tf.keras.backend.image_data_format()
        data_format_string = "NHWC" if data_format == 'channels_last' else "NCHW"

        # Calculate output shape for conv2d_transpose
        if data_format == 'channels_last': # NHWC
            h_in, w_in = input_shape[1], input_shape[2]
            output_shape = tf.stack([batch_size,
                                     self.deconv_length(h_in, self.strides, self.kernel_size, self.padding),
                                     self.deconv_length(w_in, self.strides, self.kernel_size, self.padding),
                                     self.out_ch])
            strides_list = [1, self.strides, self.strides, 1]
        else: # NCHW
            h_in, w_in = input_shape[2], input_shape[3]
            output_shape = tf.stack([batch_size,
                                     self.out_ch,
                                     self.deconv_length(h_in, self.strides, self.kernel_size, self.padding),
                                     self.deconv_length(w_in, self.strides, self.kernel_size, self.padding)])
            strides_list = [1, 1, self.strides, self.strides]

        effective_kernel = self.kernel
        if self.use_wscale and self.wscale is not None:
             # Ensure wscale dtype matches kernel dtype
             wscale_casted = tf.cast(self.wscale, self.kernel.dtype)
             effective_kernel = self.kernel * wscale_casted

        # Perform the transpose convolution
        outputs = tf.nn.conv2d_transpose(
            inputs,
            effective_kernel,
            output_shape,
            strides=strides_list,
            padding=self.padding, # Uses 'SAME' or 'VALID' string
            data_format=data_format_string)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=data_format_string)

        return outputs

    # Keep helper deconv_length (uses standard math, no nn dependency)
    def deconv_length(self, dim_size, stride_size, kernel_size, padding):
         # Ensure dim_size is int for calculations if possible
         dim_size_int = tf.cast(dim_size, tf.int32) if isinstance(dim_size, tf.Tensor) else dim_size
         padding_upper = padding.upper() # Use uppercase for comparison

         if dim_size is None: return None # Should not happen with TF2 eager/graph

         if padding_upper == 'VALID':
             output_dim = dim_size_int * stride_size + max(kernel_size - stride_size, 0)
         elif padding_upper == 'SAME':
             output_dim = dim_size_int * stride_size
         # Note: 'FULL' padding might require specific calculation based on tf.nn.conv2d_transpose docs if used
         # elif padding_upper == 'FULL':
         #     output_dim = dim_size_int * stride_size - (stride_size + kernel_size - 2)
         else:
              raise ValueError(f"Unsupported padding format: {padding}")
         return output_dim

    def get_config(self):
        # Keras standard way to save config
        config = super().get_config()
        config.update({
            'out_ch': self.out_ch,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'use_wscale': self.use_wscale,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer_arg),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer_arg),
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Keras standard way to load from config
        # Deserialize initializers
        config['kernel_initializer'] = tf.keras.initializers.deserialize(config.get('kernel_initializer'))
        config['bias_initializer'] = tf.keras.initializers.deserialize(config.get('bias_initializer'))
        return cls(**config)


# --- DO NOT ASSIGN back to nn.Conv2DTranspose here ---
# --- END OF FILE ---