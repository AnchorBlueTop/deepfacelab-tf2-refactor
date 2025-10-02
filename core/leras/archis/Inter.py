# --- START OF FILE core/leras/archis/Inter.py --- (Corrected Super Call)

import tensorflow as tf
import numpy as np
import sys

class Inter(tf.keras.Model):
    def __init__(self, ae_ch, ae_out_ch, lowest_dense_res, opts='', activation=None, use_fp16=False, name="Inter",
                 Dense_cls=None,
                 Upscale_cls=None,
                 Conv2D_cls=None,
                 **kwargs): # Keep kwargs to catch standard Keras args like 'dtype'

        # --- Call super() FIRST, passing ONLY known Keras args ---
        # Extract standard Keras args that might be in kwargs
        keras_kwargs = {
            'name': name,
            'dtype': kwargs.get('dtype', None) # Pass dtype explicitly if present
            # Add any other standard Keras Model/Layer kwargs you might use
        }
        # Remove them from kwargs so they aren't passed via **kwargs if defined explicitly
        kwargs.pop('name', None)
        kwargs.pop('dtype', None)
        if kwargs: # If other unexpected kwargs remain, raise error or warning
             print(f"Warning: Inter received unexpected kwargs: {kwargs}")
             
        super().__init__(**keras_kwargs) # Call with only known Keras args
        # -----------------------------------------------------------

        # Now process the custom arguments
        self.ae_ch = ae_ch
        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res
        self.opts = opts
        self.activation = activation # Store activation to pass to Upscale
        self.use_fp16 = use_fp16

        # --- Check and Store passed classes ---
        if Dense_cls is None or not issubclass(Dense_cls, tf.keras.layers.Layer): raise ValueError("Inter requires Dense_cls.")
        if Upscale_cls is None or not issubclass(Upscale_cls, tf.keras.layers.Layer): raise ValueError("Inter requires Upscale_cls.")
        if Conv2D_cls is None or not issubclass(Conv2D_cls, tf.keras.layers.Layer): raise ValueError("Inter requires Conv2D_cls for Upscale.")
        self.Dense_cls = Dense_cls
        self.Upscale_cls = Upscale_cls
        self.Conv2D_cls = Conv2D_cls
        # ------------------------------------

        # ... (Rest of __init__ including layer instantiation) ...
        self.dense2_units = self.lowest_dense_res * self.lowest_dense_res * self.ae_out_ch;
        self.reshape_target_shape = None; self.upscale1 = None;

        self.dense1 = self.Dense_cls(units=self.ae_ch, name="dense1")
        self.dense2 = self.Dense_cls(units=self.dense2_units, name="dense2")

        if 't' not in self.opts:
             # Use the correct parameter names for the Upscale class
             self.upscale1 = self.Upscale_cls(out_ch=self.ae_out_ch, kernel_size=3, activation=self.activation, name="upscale1",
                                           Conv2D_cls=self.Conv2D_cls) # Pass Conv2D_cls down


    def build(self, input_shape):
        """Define reshape target shape based on actual data format."""
        # Can be called multiple times, check if already defined
        if self.reshape_target_shape is None:
            # Use tf.keras.backend here instead of nn.
            if tf.keras.backend.image_data_format() == 'channels_last': # NHWC
                self.reshape_target_shape = (self.lowest_dense_res, self.lowest_dense_res, self.ae_out_ch)
            else: # channels_first (NCHW)
                self.reshape_target_shape = (self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)
        super().build(input_shape) # Ensure parent build is called


    def call(self, inputs, training=False): # Add training argument
        """Forward pass for the Inter block."""
        x = inputs
        
        # ---- DEBUG Inter.call: Input to Inter ----
        if x is not None:
            tf.print(f"DEBUG Inter ({self.name}) input (Encoder_output) min/max/mean/std:", 
                     tf.reduce_min(x), tf.reduce_max(x), 
                     tf.reduce_mean(x), tf.math.reduce_std(x), 
                     output_stream=sys.stdout, summarize=4) # summarize to see a few values
        # -----------------------------------------

        # --- Dense Layers ---
        # Pass training flag if Dense layer ever uses BN/Dropout (it doesn't currently)
        x = self.dense1(x)
        
        # ---- DEBUG Inter.call: After Dense1 ----
        if x is not None:
            tf.print(f"DEBUG Inter ({self.name}) after_dense1 min/max/mean/std:", 
                     tf.reduce_min(x), tf.reduce_max(x), 
                     tf.reduce_mean(x), tf.math.reduce_std(x), 
                     output_stream=sys.stdout, summarize=4)
        # --------------------------------------
        
        x = self.dense2(x)
        
        # ---- DEBUG Inter.call: After Dense2 ----
        if x is not None:
            tf.print(f"DEBUG Inter ({self.name}) after_dense2 min/max/mean/std:", 
                     tf.reduce_min(x), tf.reduce_max(x), 
                     tf.reduce_mean(x), tf.math.reduce_std(x), 
                     output_stream=sys.stdout, summarize=4)
        # --------------------------------------

        # --- Reshape ---
        # Ensure reshape target is defined (build should have been called)
        if self.reshape_target_shape is None:
            self.build(tf.TensorShape(inputs.shape)) # Call build if called before build

        # Reshape using tf.reshape. Target shape excludes batch dimension.
        x = tf.reshape(x, (-1,) + self.reshape_target_shape )
        
        # ---- DEBUG Inter.call: After Reshape ----
        if x is not None:
            tf.print(f"DEBUG Inter ({self.name}) after_reshape min/max/mean/std:", 
                     tf.reduce_min(x), tf.reduce_max(x), 
                     tf.reduce_mean(x), tf.math.reduce_std(x), 
                     output_stream=sys.stdout, summarize=4)
        # ----------------------------------------

        # --- Optional FP16 Cast ---
        layer_compute_dtype = tf.float16 if self.use_fp16 else tf.float32
        if x.dtype != layer_compute_dtype:
             x = tf.cast(x, layer_compute_dtype)

        # --- Conditional Upscale ---
        if self.upscale1 is not None: # Equivalent to 't' not in self.opts
             # Pass training flag if Upscale uses BN/Dropout (it doesn't currently)
             x = self.upscale1(x)

        # --- Optional Cast Back to FP32 ---
        if self.use_fp16:
             x = tf.cast(x, tf.float32)
        
        # ---- DEBUG Inter.call: Final Output from Inter ----
        if x is not None:
            tf.print(f"DEBUG Inter ({self.name}) final_output (to Decoder) min/max/mean/std:", 
                     tf.reduce_min(x), tf.reduce_max(x), 
                     tf.reduce_mean(x), tf.math.reduce_std(x), 
                     output_stream=sys.stdout, summarize=4)
        # -------------------------------------------------

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'ae_ch': self.ae_ch,
            'ae_out_ch': self.ae_out_ch,
            'lowest_dense_res': self.lowest_dense_res,
            'opts': self.opts,
            'activation_name': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation),
            'use_fp16': self.use_fp16,
            # Note: Passed classes (Dense_cls, Upscale_cls) are not easily serializable by default.
            # Saving/loading might require passing them again via custom_objects.
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Need to pop activation_name and potentially handle custom activation
        config.pop('activation_name', None)
        # IMPORTANT: Keras cannot automatically inject the Dense_cls and Upscale_cls here.
        # The code that calls from_config (usually tf.keras.models.load_model)
        # needs to provide these classes in the `custom_objects` dictionary.
        # Example: load_model(..., custom_objects={'Dense': Dense, 'Upscale': Upscale, 'Inter': Inter})
        return cls(**config)


# --- REMOVE nn.Inter = Inter ---
# --- END OF FILE core/leras/archis/Inter.py ---