# --- START OF FILE core/leras/archis/Encoder.py --- (Corrected Super Call)

import tensorflow as tf
from tensorflow.keras.layers import Flatten # Keep standard Keras imports
import numpy as np
import sys

class Encoder(tf.keras.Model):
    def __init__(self, e_ch, opts='', activation=None, use_fp16=False, name="Encoder",
                 # --- Accept Layer CLASSES as arguments ---
                 Downscale_cls=None,
                 DownscaleBlock_cls=None,
                 ResidualBlock_cls=None,
                 Conv2D_cls=None, # Added
                 gradient_checkpointing=False, # Add gradient_checkpointing flag
                 **kwargs):

        # --- Call super() FIRST, passing ONLY known Keras args ---
        keras_kwargs = {
            'name': name,
            'dtype': kwargs.get('dtype', None) # Pass dtype explicitly if present
        }
        # Remove standard Keras args from kwargs to avoid passing duplicates
        kwargs.pop('name', None)
        kwargs.pop('dtype', None)
        if kwargs: # If other unexpected kwargs remain, raise error or warning
             print(f"Warning: Encoder received unexpected kwargs: {kwargs}")

        super().__init__(**keras_kwargs) # Call with only known Keras args
        # -----------------------------------------------------------

        # Now process the custom arguments
        self.e_ch = e_ch
        self.opts = opts
        self.activation = activation
        self.use_fp16 = use_fp16
        self.conv_dtype = tf.float16 if use_fp16 else (self.dtype or tf.keras.backend.floatx())
        self.gradient_checkpointing = gradient_checkpointing  # Store gradient_checkpointing flag

        # Define input channels
        input_ch = 3 # Define input channels

        # --- Check if required classes were actually passed ---
        if Downscale_cls is None or DownscaleBlock_cls is None or \
           ResidualBlock_cls is None or Conv2D_cls is None:
            raise ValueError("Encoder requires Downscale_cls, DownscaleBlock_cls, ResidualBlock_cls, and Conv2D_cls classes.")

        # --- Optional: Verify passed classes ---
        if not issubclass(Downscale_cls, tf.keras.layers.Layer) or \
           not issubclass(DownscaleBlock_cls, tf.keras.layers.Layer) or \
           not issubclass(ResidualBlock_cls, tf.keras.layers.Layer) or \
           not issubclass(Conv2D_cls, tf.keras.layers.Layer):
            print("Warning: Encoder received non-Keras Layer class arguments.")

        # --- Define layers using PASSED class arguments, passing down base classes ---
        if 't' in self.opts:
            self.down1 = Downscale_cls(in_ch=3, out_ch=self.e_ch, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_1", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.res1 = ResidualBlock_cls(ch=self.e_ch, kernel_size=3, activation=self.activation, use_fp16=self.use_fp16, name="res_1", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.down2 = Downscale_cls(in_ch=self.e_ch, out_ch=self.e_ch*2, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_2", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.down3 = Downscale_cls(in_ch=self.e_ch*2, out_ch=self.e_ch*4, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_3", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.down4 = Downscale_cls(in_ch=self.e_ch*4, out_ch=self.e_ch*8, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_4", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.down5 = Downscale_cls(in_ch=self.e_ch*8, out_ch=self.e_ch*8, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_5", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
            self.res5 = ResidualBlock_cls(ch=self.e_ch*8, kernel_size=3, activation=self.activation, use_fp16=self.use_fp16, name="res_5", dtype=self.conv_dtype, Conv2D_cls=Conv2D_cls)
        else:
            self.down_block = DownscaleBlock_cls(input_ch=3, ch=self.e_ch, n_downscales=4, kernel_size=5, activation=self.activation, use_fp16=self.use_fp16, name="down_block", dtype=self.conv_dtype,
                                              Downscale_cls=Downscale_cls, # Pass Downscale_cls
                                              Conv2D_cls=Conv2D_cls)       # Pass Conv2D_cls

        self.flatten_layer = Flatten(name="flatten", dtype=self.compute_dtype)

    # --- call method remains the same ---
    def call(self, inputs, training=False):
        # --- Checkpointing Logic REMOVED from Encoder.call ---

        x = inputs;
        if self.use_fp16: x = tf.cast(x, tf.float16)

        if 't' in self.opts:
             # Directly call layers
             x = self.down1(x);
             x = self.res1(x, training=training);
             x = self.down2(x);
             x = self.down3(x);
             x = self.down4(x);
             x = self.down5(x);
             x = self.res5(x, training=training)
        else:
             # Directly call block
             x = self.down_block(x, training=training)

        x = self.flatten_layer(x);
        
        # ---- DEBUG Encoder before Pixel Norm ----
        if x is not None: # Add check for None
            tf.print(f"DEBUG Encoder ({self.name}) BEFORE PixelNorm: x min/max/mean/std:",
                     tf.reduce_min(x), tf.reduce_max(x),
                     tf.reduce_mean(x), tf.math.reduce_std(x),
                     output_stream=sys.stdout, summarize=4)
        # -----------------------------------------
        
        # -- TEMPORARILY DISABLED PIXEL NORMALIZATION --
        # if 'u' in self.opts:
        #      tf.print(f"DEBUG Encoder ({self.name}): Applying Pixel Normalization ('u' in opts)", output_stream=sys.stdout)
        #      epsilon = tf.keras.backend.epsilon()
        #      x = x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + epsilon)
        #      
        #      # ---- DEBUG Encoder AFTER Pixel Norm ----
        #      if x is not None: # Add check for None
        #          tf.print(f"DEBUG Encoder ({self.name}) AFTER PixelNorm: x min/max/mean/std:",
        #                   tf.reduce_min(x), tf.reduce_max(x),
        #                   tf.reduce_mean(x), tf.math.reduce_std(x),
        #                   output_stream=sys.stdout, summarize=4)
        #      # ----------------------------------------
        
        # Add a print to confirm PN is skipped
        tf.print(f"DEBUG Encoder ({self.name}): Pixel Normalization step SKIPPED for this test.", output_stream=sys.stdout)
        if self.use_fp16: x = tf.cast(x, tf.float32) # Ensure cast back if fp16 was used
        return x

    # --- get_config / from_config need updating for Conv2D_cls ---
    def get_config(self):
        config = super().get_config()
        config.update({
            'e_ch': self.e_ch,
            'opts': self.opts,
            'activation_name': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation),
            'use_fp16': self.use_fp16,
            'input_ch': 3,
            'gradient_checkpointing': self.gradient_checkpointing,
            'dtype': tf.keras.mixed_precision.dtype_policy().name if isinstance(self.dtype_policy, tf.keras.mixed_precision.Policy) else self.dtype
        })
        return config
    @classmethod
    def from_config(cls, config, custom_objects=None):
        if custom_objects is None or 'Downscale_cls' not in custom_objects or 'DownscaleBlock_cls' not in custom_objects or 'ResidualBlock_cls' not in custom_objects or 'Conv2D_cls' not in custom_objects: # Added Conv2D_cls check
            raise ValueError("from_config for Encoder requires Downscale_cls, DownscaleBlock_cls, ResidualBlock_cls, and Conv2D_cls in custom_objects")
        config['Downscale_cls'] = custom_objects['Downscale_cls']; config['DownscaleBlock_cls'] = custom_objects['DownscaleBlock_cls']; config['ResidualBlock_cls'] = custom_objects['ResidualBlock_cls']; config['Conv2D_cls'] = custom_objects['Conv2D_cls']; # Added Conv2D_cls
        config.pop('activation_name', None); return cls(**config)

# --- END OF FILE core/leras/archis/Encoder.py ---