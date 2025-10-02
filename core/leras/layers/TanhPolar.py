# --- START OF FILE core/leras/layers/TanhPolar.py --- (Refactored)

import numpy as np
import tensorflow as tf
# Need tensorflow-addons for resampler, or implement manually
try:
    import tensorflow_addons as tfa
    print("TensorFlow Addons found. Using tfa.image.resampler.")
    bilinear_sampler_func = tfa.image.resampler
except ImportError:
    print("Warning: TensorFlow Addons not found. TanhPolar requires tfa.image.resampler or a manual implementation.")
    # Define a dummy function or raise error if essential
    def bilinear_sampler_func(img, gridx, gridy):
         raise NotImplementedError("TanhPolar requires tfa.image.resampler or manual implementation.")

# Inherit from Keras Layer, although it has no weights
class TanhPolar(tf.keras.layers.Layer):
    """
    Tanh-polar Transformer Network Layer (Keras implementation).
    Provides warp() and restore() methods based on precomputed grids.
    Requires tensorflow_addons for bilinear sampling (tfa.image.resampler).
    """
    def __init__(self, width, height, angular_offset_deg=270, name=None, dtype=None, **kwargs):
        """
        Args:
            width: Integer, width of the polar space/output grid.
            height: Integer, height of the polar space/output grid.
            angular_offset_deg: Float, angular offset in degrees.
            name: Layer name.
            dtype: Layer dtype (used for constants).
            **kwargs: Additional Layer arguments.
        """
        super().__init__(name=name, dtype=dtype, trainable=False, **kwargs) # Not trainable
        self.grid_width = width
        self.grid_height = height
        self.angular_offset_deg = angular_offset_deg

        # Use layer's dtype for constants
        layer_dtype = self.dtype or tf.keras.backend.floatx()
        layer_dtype_tf = tf.dtypes.as_dtype(layer_dtype) # Ensure TF DType object

        # Precompute grids and store as non-trainable constants/variables
        warp_gridx, warp_gridy = TanhPolar._get_tanh_polar_warp_grids(
            self.grid_width, self.grid_height, self.angular_offset_deg
        )
        restore_gridx, restore_gridy = TanhPolar._get_tanh_polar_restore_grids(
            self.grid_width, self.grid_height, self.angular_offset_deg
        )

        # Use tf.constant for non-trainable grid data
        # Expand dims for batch compatibility (add batch dimension of 1)
        self.warp_gridx_t = tf.constant(warp_gridx[np.newaxis, ...], dtype=layer_dtype_tf)
        self.warp_gridy_t = tf.constant(warp_gridy[np.newaxis, ...], dtype=layer_dtype_tf)
        self.restore_gridx_t = tf.constant(restore_gridx[np.newaxis, ...], dtype=layer_dtype_tf)
        self.restore_gridy_t = tf.constant(restore_gridy[np.newaxis, ...], dtype=layer_dtype_tf)

    # No weights to build
    # def build(self, input_shape):
    #     super().build(input_shape)

    # This layer doesn't have a standard 'call', provide warp/restore methods
    def warp(self, inp_t):
        """Applies forward Tanh-Polar warp."""
        batch_t = tf.shape(inp_t)[0]
        # Tile grids to match batch size
        warp_gridx_t = tf.tile(self.warp_gridx_t, (batch_t, 1, 1))
        warp_gridy_t = tf.tile(self.warp_gridy_t, (batch_t, 1, 1))
        # Stack grids for resampler: shape (batch, H, W, 2) with (x, y) coordinates
        warp_sampling_grid = tf.stack([warp_gridx_t, warp_gridy_t], axis=-1)

        # Ensure input is float
        inp_t_float = tf.cast(inp_t, self.dtype or tf.keras.backend.floatx())

        # Handle data format - resampler usually expects NHWC
        data_format = tf.keras.backend.image_data_format()
        inp_nhwc = inp_t_float
        if data_format == 'channels_first':
            inp_nhwc = tf.transpose(inp_t_float, (0, 2, 3, 1))

        # Use the bilinear sampler function (e.g., tfa.image.resampler)
        out_nhwc = bilinear_sampler_func(inp_nhwc, warp_sampling_grid)

        # Transpose back if needed
        out_t = out_nhwc
        if data_format == 'channels_first':
            out_t = tf.transpose(out_nhwc, (0, 3, 1, 2))

        return out_t

    def restore(self, inp_t):
        """Applies inverse Tanh-Polar warp (restore)."""
        batch_t = tf.shape(inp_t)[0]
        # Tile grids
        restore_gridx_t = tf.tile(self.restore_gridx_t, (batch_t, 1, 1))
        restore_gridy_t = tf.tile(self.restore_gridy_t, (batch_t, 1, 1))
        # Stack grids for resampler: shape (batch, H, W, 2) with (x, y) coordinates
        restore_sampling_grid = tf.stack([restore_gridx_t, restore_gridy_t], axis=-1)

        # Ensure input is float
        inp_t_float = tf.cast(inp_t, self.dtype or tf.keras.backend.floatx())

        # Handle data format - resampler expects NHWC
        data_format = tf.keras.backend.image_data_format()
        inp_nhwc = inp_t_float
        if data_format == 'channels_first':
            inp_nhwc = tf.transpose(inp_t_float, (0, 2, 3, 1))

        # Symmetric padding as in original code
        # Padding is applied to H and W dimensions (axes 1 and 2 for NHWC)
        inp_padded_nhwc = tf.pad(inp_nhwc, [(0,0), (1, 1), (1, 0), (0, 0)], "SYMMETRIC")

        # Use the bilinear sampler function
        out_nhwc = bilinear_sampler_func(inp_padded_nhwc, restore_sampling_grid)

        # Transpose back if needed
        out_t = out_nhwc
        if data_format == 'channels_first':
            out_t = tf.transpose(out_nhwc, (0, 3, 1, 2))

        return out_t

    # Static methods for grid calculation (no TF ops, uses NumPy - OK)
    @staticmethod
    def _get_tanh_polar_warp_grids(W, H, angular_offset_deg):
        # ... (Keep original NumPy implementation) ...
        angular_offset_pi = angular_offset_deg * np.pi / 180.0
        roi_center = np.array([ W//2, H//2], np.float32 ); roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5; cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi);
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W),np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1); radii = normalised_dest_indices[..., 0];
        orientation_x = np.cos(normalised_dest_indices[..., 1]); orientation_y = np.sin(normalised_dest_indices[..., 1]);
        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2));
        src_x_indices = src_radii * orientation_x; src_y_indices = src_radii * orientation_y;
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices, roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices);
        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W, H, angular_offset_deg):
        # ... (Keep original NumPy implementation) ...
        angular_offset_pi = angular_offset_deg * np.pi / 180.0; roi_center = np.array([ W//2, H//2], np.float32 ); roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5; cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi);
        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float); normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],[sin_offset, cos_offset]]));
        radii = np.linalg.norm(normalised_dest_indices, axis=-1); clip_radii = np.clip(radii, 1e-9, None); normalised_dest_indices[..., 0] /= clip_radii; normalised_dest_indices[..., 1] /= clip_radii;
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 + roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1];
        src_radii = np.tanh(radii); src_x_indices = src_radii * W + 1.0; src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) / 2.0 / np.pi) * H, H) + 1.0;
        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'width': self.grid_width,
            'height': self.grid_height,
            'angular_offset_deg': self.angular_offset_deg,
        })
        return config

# --- REMOVE assignment back ---
# nn.TanhPolar = TanhPolar
# --- END OF FILE ---