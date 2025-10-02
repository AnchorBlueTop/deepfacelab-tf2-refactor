# --- START OF FILE core/leras/losses/DssimLoss.py ---
# (Recommended to move to a new 'losses' subdirectory)

import tensorflow as tf
import numpy as np
from core.leras import nn # Keep for nn.data_format

class DssimLoss(tf.keras.losses.Loss):
    """
    DSSIM Loss Function (1.0 - SSIM) / 2.0

    Inherits from tf.keras.losses.Loss for Keras integration.
    Uses depthwise convolution for SSIM calculation.
    """
    def __init__(self,
                 max_val=1.0,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 name='dssim_loss',
                 **kwargs):
        """
        Args:
            max_val: Float. The maximum possible pixel value (usually 1.0).
            filter_size: Integer. Size of the Gaussian filter kernel.
            filter_sigma: Float. Sigma for the Gaussian filter kernel.
            k1: Float. K1 constant for SSIM calculation.
            k2: Float. K2 constant for SSIM calculation.
            name: String. Name for the loss instance.
            **kwargs: Additional arguments for Loss base class.
        """
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.max_val = max_val
        self.filter_size = max(1, filter_size)
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

        # Precompute Gaussian kernel (TF constant)
        self.kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        """Creates the Gaussian kernel as a TF constant."""
        kernel_1d = np.arange(0, self.filter_size, dtype=np.float32)
        kernel_1d -= (self.filter_size - 1) / 2.0
        kernel_1d = kernel_1d**2
        kernel_1d *= (-0.5 / (self.filter_sigma**2))
        # Apply exp after calculating exponent values
        kernel_1d = np.exp(kernel_1d)

        # Create 2D kernel and normalize
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= np.sum(kernel_2d)

        # Reshape for depthwise_conv2d: [height, width, in_channels=1, channel_multiplier=1]
        kernel_tf = tf.constant(kernel_2d[:, :, np.newaxis, np.newaxis], dtype=tf.float32)
        return kernel_tf


    def call(self, y_true, y_pred):
        """
        Calculates the DSSIM loss.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Loss tensor (scalar DSSIM value averaged over batch).
        """
        if y_true.dtype != y_pred.dtype:
             # Keras usually handles this, but ensure consistency
             y_pred = tf.cast(y_pred, y_true.dtype)

        # Use float32 for calculations, cast back if needed (though loss is usually float32)
        y_true_f32 = tf.cast(y_true, tf.float32)
        y_pred_f32 = tf.cast(y_pred, tf.float32)

        # Get channel count and tile kernel
        if nn.data_format == 'channels_first': # NCHW
            num_channels = tf.shape(y_true_f32)[1]
            y_true_f32 = tf.transpose(y_true_f32, [0, 2, 3, 1]) # Temp transpose to NHWC for conv
            y_pred_f32 = tf.transpose(y_pred_f32, [0, 2, 3, 1]) # Temp transpose to NHWC for conv
            kernel_ready = tf.tile(self.kernel, [1, 1, num_channels, 1])
            current_data_format = "NHWC" # Use NHWC for conv op
            spatial_axes = [1, 2] # Axes H, W for NHWC
        else: # NHWC
            num_channels = tf.shape(y_true_f32)[-1]
            kernel_ready = tf.tile(self.kernel, [1, 1, num_channels, 1])
            current_data_format = "NHWC"
            spatial_axes = [1, 2] # Axes H, W

        # Helper function for depthwise convolution (local mean/variance)
        def reducer(x):
            # Padding='VALID' because we handle kernel application like a filter window
            return tf.nn.depthwise_conv2d(x, kernel_ready, strides=[1, 1, 1, 1],
                                          padding='VALID', data_format=current_data_format)

        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # Calculate means
        mean0 = reducer(y_true_f32)
        mean1 = reducer(y_pred_f32)

        # Calculate luminance component
        num0 = mean0 * mean1 * 2.0
        den0 = tf.square(mean0) + tf.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        # Calculate contrast/structure component
        num1 = reducer(y_true_f32 * y_pred_f32) * 2.0
        den1 = reducer(tf.square(y_true_f32) + tf.square(y_pred_f32))
        # c2 *= 1.0 # Original comment, seems redundant
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        # Combine components for SSIM
        # Average SSIM over spatial dimensions for each image/channel
        ssim_val_per_channel = tf.reduce_mean(luminance * cs, axis=spatial_axes)

        # Average SSIM across channels for each image in the batch
        ssim_val_per_image = tf.reduce_mean(ssim_val_per_channel, axis=-1) # Average over channel axis

        # Calculate DSSIM loss = (1 - SSIM) / 2
        # Average DSSIM over the batch
        dssim_loss = tf.reduce_mean((1.0 - ssim_val_per_image) / 2.0)

        return dssim_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_val': self.max_val,
            'filter_size': self.filter_size,
            'filter_sigma': self.filter_sigma,
            'k1': self.k1,
            'k2': self.k2,
        })
        return config

# Assign the refactored class back to nn.dssim (or a new name like nn.DssimLoss)
# nn.DssimLoss = DssimLoss
lambda img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.02: \
DssimLoss(max_val, filter_size, filter_sigma, k1, k2)(img1, img2) # Lambda for compatibility
# --- END OF FILE core/leras/losses/DssimLoss.py ---

# Assign the refactored class back to nn.dssim (or a new name like nn.DssimLoss)
# nn.DssimLoss = DssimLoss # This assignment belongs in nn.py's import helper

# --- END OF FILE ---