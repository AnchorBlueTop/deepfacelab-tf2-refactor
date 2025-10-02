# --- START OF FILE core/leras/layers/BlurPool.py --- (Refactored)

import numpy as np
# from core.leras import nn # REMOVE
import tensorflow as tf

# Inherit from Keras Layer
class BlurPool(tf.keras.layers.Layer):
    """
    BlurPooling layer based on anti-aliased conv principles (Keras implementation).
    Uses a fixed, predefined blur kernel applied via depthwise convolution.
    """
    def __init__(self, filt_size=3, stride=2, name=None, dtype=None, **kwargs):
        """
        Args:
            filt_size: Size of the blur kernel (1 to 7).
            stride: Downsampling stride (usually 2).
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        super().__init__(name=name, dtype=dtype, trainable=False, **kwargs) # Not trainable
        self.filt_size = filt_size
        self.stride = stride

        # Precompute kernel array (NumPy)
        self.np_kernel = self._create_blur_kernel()
        # Kernel tensor (self.k) will be created in build()

    def _create_blur_kernel(self):
        """Creates the NumPy blur kernel based on filter size."""
        if self.filt_size == 1: a = np.array([1.,])
        elif self.filt_size == 2: a = np.array([1., 1.])
        elif self.filt_size == 3: a = np.array([1., 2., 1.])
        elif self.filt_size == 4: a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5: a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6: a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7: a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else: raise ValueError(f"Unsupported BlurPool filter size: {self.filt_size}")

        a = a[:, None] * a[None, :] # Create 2D kernel
        a = a / np.sum(a) # Normalize
        # Reshape for depthwise_conv2d: [H, W, In_C=1, DepthMultiplier=1]
        a = a[:, :, np.newaxis, np.newaxis]
        return a.astype(np.float32) # Use float32 for kernel

    def build(self, input_shape):
        """Creates the kernel Tensor."""
        if not self.built:
            layer_dtype = self.dtype or tf.keras.backend.floatx()
            # Create non-trainable constant Tensor for the kernel
            self.k = tf.constant(self.np_kernel, dtype=layer_dtype)
        super().build(input_shape)

    def call(self, x):
        """Applies blur pooling."""
        # Get data format and determine padding/strides
        data_format = tf.keras.backend.image_data_format()
        data_format_string = "NHWC" if data_format == 'channels_last' else "NCHW"

        # Determine stride list
        stride_list = [1, self.stride, self.stride, 1] if data_format == 'channels_last' else [1, 1, self.stride, self.stride]

        # Determine padding list for manual tf.pad
        # tf.nn.depthwise_conv2d with 'VALID' expects manual padding for blurring
        pad_total = self.filt_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad = [pad_beg, pad_end]
        padding_list = [[0,0], pad, pad, [0,0]] if data_format == 'channels_last' else [[0,0], [0,0], pad, pad]

        # Get input channel count dynamically
        channel_axis = -1 if data_format == 'channels_last' else 1
        in_ch = tf.shape(x)[channel_axis]

        # Tile the kernel to match input channels for depthwise conv
        # Kernel shape: [H, W, In_C=1, DepthMultiplier=1]
        # Tiled shape:  [H, W, In_C, DepthMultiplier=1]
        kernel_ready = tf.tile(self.k, (1, 1, in_ch, 1))

        # Pad the input
        x_padded = tf.pad(x, padding_list, mode='REFLECT') # Use REFLECT or SYMMETRIC padding? Original didn't specify.

        # Apply depthwise convolution as the blur + downsample operation
        pooled = tf.nn.depthwise_conv2d(
                    x_padded,
                    kernel_ready,
                    strides=stride_list,
                    padding='VALID', # Padding was done manually
                    data_format=data_format_string
                 )
        return pooled

    def get_config(self):
        config = super().get_config()
        config.update({
            'filt_size': self.filt_size,
            'stride': self.stride,
        })
        return config

# --- DO NOT ASSIGN back to nn.BlurPool here ---
# --- END OF FILE ---