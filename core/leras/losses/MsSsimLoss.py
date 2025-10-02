# --- START OF FILE core/leras/losses/MsSsimLoss.py ---
# (Recommended to move to a new 'losses' subdirectory)

import tensorflow as tf
from core.leras import nn # Keep for nn.get_gaussian_weights for now

# Inherit from Keras Loss base class
class MsSsimLoss(tf.keras.losses.Loss):
    """
    Multi-Scale SSIM Loss, optionally combined with L1 loss.

    Inherits from tf.keras.losses.Loss for better Keras integration.
    """
    default_power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    default_l1_alpha = 0.84

    def __init__(self,
                 resolution, # Required to calculate valid power factors
                 kernel_size=11,
                 max_val=1.0, # Max value of the input images (usually 1.0 for float)
                 use_l1=False,
                 l1_alpha=None, # Optional override for alpha
                 power_factors=None, # Optional override for power factors
                 name='ms_ssim_loss',
                 **kwargs):
        """
        Args:
            resolution: Integer, the resolution of the input images (e.g., 256).
                        Used to determine valid scales.
            kernel_size: Integer, the size of the Gaussian filter kernel used in SSIM.
            max_val: Float, the maximum possible pixel value.
            use_l1: Boolean, whether to combine MS-SSIM with L1 loss.
            l1_alpha: Float, weighting factor for MS-SSIM when use_l1 is True.
                      Defaults to 0.84.
            power_factors: Tuple/List of floats. Weights for each scale. Defaults
                           to standard MS-SSIM factors, adjusted for resolution.
            name: String, name for the loss instance.
            **kwargs: Additional arguments for Loss base class (e.g., reduction).
        """
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs) # Pass reduction, name etc. to parent

        self.kernel_size = kernel_size
        self.max_val = max_val
        self.use_l1 = use_l1
        self.l1_alpha = l1_alpha if l1_alpha is not None else self.default_l1_alpha

        # Determine and store power factors
        if power_factors is None:
            pf = [p for i, p in enumerate(self.default_power_factors) if resolution//(2**i) >= kernel_size]
            # Normalize if needed
            if sum(pf) < 1.0 and sum(pf) > 0: # Avoid division by zero if all scales too small
                self.power_factors = [x / sum(pf) for x in pf]
            else:
                 self.power_factors = pf # Use as is or empty if resolution too low
        else:
            self.power_factors = power_factors

        self.num_scale = len(self.power_factors)

        # Store gaussian weights if L1 is used
        # Note: This requires batch_size and in_ch, which are typically not known
        # when the loss is initialized. This is a problem with the original design.
        # Option 1: Create weights inside call() (less efficient).
        # Option 2: Assume fixed batch_size/in_ch (less flexible).
        # Option 3: Refactor get_gaussian_weights to not need bs/in_ch, or pass dynamically.
        # For now, we'll defer the creation of gaussian_weights to the call() method.
        self.gaussian_weights = None
        # If you know bs/in_ch at init, you could create them here:
        # if use_l1:
        #     # self.gaussian_weights = nn.get_gaussian_weights(batch_size, in_ch, resolution, num_scale=self.num_scale) # Needs bs, in_ch
        #     pass

    # Keras Loss primary method is call()
    def call(self, y_true, y_pred):
        """
        Calculates the MS-SSIM loss (optionally combined with L1).

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Loss tensor.
        """
        # Ensure inputs are float32, as required by tf.image.ssim_multiscale
        y_true_f32 = tf.cast(y_true, tf.float32)
        y_pred_f32 = tf.cast(y_pred, tf.float32)

        # Determine data format and transpose if necessary for tf.image.ssim_multiscale (expects NHWC)
        data_format = tf.keras.backend.image_data_format() # Get Keras backend format
        if data_format == 'channels_first': # NCHW
            y_true_t = tf.transpose(y_true_f32, [0, 2, 3, 1])
            y_pred_t = tf.transpose(y_pred_f32, [0, 2, 3, 1])
        else: # NHWC
            y_true_t = y_true_f32
            y_pred_t = y_pred_f32

        # Handle case where resolution might be too low for any power factors
        if not self.power_factors:
             print("Warning: Input resolution likely too low for MS-SSIM kernel size. Returning L1 loss.")
             # Fallback to simple L1 might be reasonable here
             return tf.reduce_mean(tf.abs(y_true_f32 - y_pred_f32))


        # Calculate MS-SSIM loss (1.0 - similarity)
        ms_ssim_val = tf.image.ssim_multiscale(
            y_true_t,
            y_pred_t,
            max_val=self.max_val,
            power_factors=self.power_factors,
            filter_size=self.kernel_size
        )
        ms_ssim_loss = 1.0 - ms_ssim_val

        # Optional L1 combination
        if self.use_l1:
            # Dynamically create or get gaussian weights here if not done in init
            if self.gaussian_weights is None:
                # Infer batch_size, in_ch, resolution from y_true
                shape = tf.shape(y_true_f32)
                bs = shape[0]
                res = shape[2] if data_format == 'channels_first' else shape[1]
                in_ch = shape[1] if data_format == 'channels_first' else shape[3]
                # Assuming nn.get_gaussian_weights is TF2 compatible:
                self.gaussian_weights = nn.get_gaussian_weights(bs, in_ch, res, num_scale=self.num_scale)
                if self.gaussian_weights is None:
                     # Handle error if get_gaussian_weights fails or needs refactoring
                     print("Error: Failed to get Gaussian weights for L1 loss calculation.")
                     return ms_ssim_loss # Fallback to just MS-SSIM

            # Calculate L1 difference (use original format y_true_f32, y_pred_f32)
            diff = tf.abs(y_true_f32 - y_pred_f32)
            # Ensure gaussian_weights match the data format for multiplication
            gaussian_weights_aligned = self.gaussian_weights
            if data_format == 'channels_last': # NHWC
                 # Weights shape might be (num_scale, bs, H, W, C) - need to adjust for NCHW diff calculation?
                 # Let's assume get_gaussian_weights returns in NCHW format for now, needs verification.
                 # If weights are NCHW (scale, bs, C, H, W), align diff (NCHW) and weights
                 if diff.shape[1] != gaussian_weights_aligned.shape[2]: # Check channel dimension
                      diff = tf.transpose(diff, [0, 3, 1, 2]) # Transpose diff to NCHW if it's NHWC
            elif data_format == 'channels_first': # NCHW
                 if diff.shape[3] != gaussian_weights_aligned.shape[4]: # Check channel dimension for NHWC weights
                      gaussian_weights_aligned = tf.transpose(gaussian_weights_aligned, [0, 1, 3, 4, 2]) # Transpose weights if they are NHWC

            # Expand dims of diff to match scales dim of weights
            diff_expanded = tf.tile(tf.expand_dims(diff, axis=0), multiples=[self.num_scale, 1, 1, 1, 1])

            # Apply weights and calculate L1 loss
            # Assuming NCHW format for weights and diff now: (scale, bs, C, H, W)
            weighted_diff = gaussian_weights_aligned[-1] * diff_expanded[-1] # Use last scale weights? Original used [-1]
            l1_loss_per_sample = tf.reduce_sum(weighted_diff, axis=[1, 2, 3]) # Sum over C, H, W
            l1_loss = tf.reduce_mean(l1_loss_per_sample) # Average over batch

            # Combine losses
            combined_loss = self.l1_alpha * ms_ssim_loss + (1.0 - self.l1_alpha) * l1_loss
            return combined_loss
        else:
            return ms_ssim_loss # Return only MS-SSIM loss


    def get_config(self):
        # Include necessary parameters for serialization
        config = super().get_config()
        # Resolution is needed to reconstruct power_factors if not passed explicitly
        # config['resolution'] = self.resolution # Need to store this if not reconstructable
        config.update({
            'kernel_size': self.kernel_size,
            'max_val': self.max_val,
            'use_l1': self.use_l1,
            'l1_alpha': self.l1_alpha,
            'power_factors': self.power_factors, # Store the calculated factors
        })
        return config

# Assign the refactored class back to nn.MsSsim for potential compatibility
# nn.MsSsimLoss = MsSsimLoss # Assign to a new name to indicate it's a Loss class
# --- END OF FILE core/leras/losses/MsSsimLoss.py ---