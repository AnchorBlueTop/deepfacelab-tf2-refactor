# --- START OF FILE core/leras/archis/Decoder.py --- (Corrected Super Call)

import tensorflow as tf
import sys
from functools import partial
# Removed: from tensorflow.keras.layers import Conv2D as StandardKerasConv2D

class Decoder(tf.keras.Model): # Inherit directly from Keras Model
    def __init__(self, d_ch, d_mask_ch, opts='', activation=None, use_fp16=False, name="Decoder",
                 # --- Accept Layer CLASSES ---
                 Conv2D_cls=None,
                 Upscale_cls=None,
                 ResidualBlock_cls=None,
                 gradient_checkpointing=False,
                 **kwargs):

        # --- Call super() FIRST, passing ONLY known Keras args ---
        keras_kwargs = {
            'name': name,
            'dtype': kwargs.get('dtype', None)
        }
        kwargs.pop('name', None)
        kwargs.pop('dtype', None)
        if kwargs: print(f"Warning: Decoder received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------------------------------------------------

        # Now process custom arguments
        self.d_ch = d_ch
        self.d_mask_ch = d_mask_ch
        self.opts = opts
        self.activation = activation # Store activation directly
        self.use_fp16 = use_fp16
        self.conv_dtype = tf.float16 if use_fp16 else (self.dtype or tf.keras.backend.floatx())
        self.gradient_checkpointing = gradient_checkpointing

        # --- Check and Store passed classes ---
        if Conv2D_cls is None or not issubclass(Conv2D_cls, tf.keras.layers.Layer): raise ValueError("Decoder requires Conv2D_cls.")
        if Upscale_cls is None or not issubclass(Upscale_cls, tf.keras.layers.Layer): raise ValueError("Decoder requires Upscale_cls.")
        if ResidualBlock_cls is None or not issubclass(ResidualBlock_cls, tf.keras.layers.Layer): raise ValueError("Decoder requires ResidualBlock_cls.")
        self.Conv2D_cls = Conv2D_cls
        self.Upscale_cls = Upscale_cls
        self.ResidualBlock_cls = ResidualBlock_cls
        # ------------------------------------

        ch_muls_t = [8, 8, 4, 2] if 't' in opts else [8, 4, 2]
        ch_muls = [8, 4, 2]

        # --- Instantiate Main Path Layers using stored Class Names ---
        self.upscales = []
        self.res_blocks = []
        last_ch = None
        for i, mul in enumerate(ch_muls_t if 't' in opts else ch_muls):
             current_out_ch = self.d_ch * mul
             # Pass Conv2D_cls down to Upscale and ResidualBlock
             self.upscales.append(self.Upscale_cls(out_ch=current_out_ch, kernel_size=3, activation=self.activation, use_fp16=self.use_fp16, name=f"upscale_{i}", Conv2D_cls=self.Conv2D_cls))
             self.res_blocks.append(self.ResidualBlock_cls(ch=current_out_ch, kernel_size=3, activation=self.activation, use_fp16=self.use_fp16, name=f"res_{i}", Conv2D_cls=self.Conv2D_cls))
             last_ch = current_out_ch

        # --- Use WScaleConv2D with small gain for output convs ---
        output_conv_gain = 1.0 # Experiment with this value
        tf.print(f"DEBUG Decoder ({self.name}): Using custom small gain for output convs: {output_conv_gain}", output_stream=sys.stdout)

        self.out_conv = self.Conv2D_cls(filters=3, kernel_size=1, padding='SAME', name="out_conv_wscale_custom_gain", gain=output_conv_gain, dtype='float32')
        if 'd' in opts:
             self.out_conv1 = self.Conv2D_cls(filters=3, kernel_size=3, padding='SAME', name="out_conv1_wscale_custom_gain", gain=output_conv_gain, dtype='float32')
             self.out_conv2 = self.Conv2D_cls(filters=3, kernel_size=3, padding='SAME', name="out_conv2_wscale_custom_gain", gain=output_conv_gain, dtype='float32')
             self.out_conv3 = self.Conv2D_cls(filters=3, kernel_size=3, padding='SAME', name="out_conv3_wscale_custom_gain", gain=output_conv_gain, dtype='float32')
        print(f"DEBUG Decoder: Built OUT_CONV with WScaleConv2D (gain={output_conv_gain}) instead of StandardKerasConv2D")

        # --- Instantiate Mask Path Layers ---
        self.upscalems = []
        num_mask_upscales = 3
        if 't' in opts: num_mask_upscales += 1
        if 'd' in opts: num_mask_upscales += 1
        last_mask_ch = None
        for i in range(num_mask_upscales):
             if i == 0: current_mask_out_ch = self.d_mask_ch * 8
             elif i == 1: current_mask_out_ch = self.d_mask_ch * (8 if 't' in opts else 4)
             elif i == 2: current_mask_out_ch = self.d_mask_ch * 4
             elif i == 3: current_mask_out_ch = self.d_mask_ch * 2
             else: current_mask_out_ch = self.d_mask_ch * 1
             # Pass Conv2D_cls down to Upscale
             self.upscalems.append(self.Upscale_cls(out_ch=current_mask_out_ch, kernel_size=3, activation=self.activation, use_fp16=self.use_fp16, name=f"upscalem_{i}", Conv2D_cls=self.Conv2D_cls))
             last_mask_ch = current_mask_out_ch

        # --- Use WScaleConv2D with small gain for mask output conv ---
        self.out_convm = self.Conv2D_cls(filters=1, kernel_size=1, padding='SAME', name="out_convm_wscale_custom_gain", gain=output_conv_gain, dtype='float32')
        print(f"DEBUG Decoder Mask: Built OUT_CONVM with WScaleConv2D (gain={output_conv_gain}) instead of StandardKerasConv2D")
    # --- End of __init__ ---

    def call(self, inputs, training=False):
        # --- Checkpointing Logic REMOVED from Decoder.call ---

        z = inputs

        # --- Main Color Path (NO Checkpointing Logic) ---
        x = z
        num_main_blocks = len(self.res_blocks)
        for i in range(num_main_blocks):
            # Call layers directly
            x = self.upscales[i](x)
            x = self.res_blocks[i](x, training=training)
        # -------------------------------------------------

        # --- Final Color Convolutions ---
        if 'd' in self.opts:
             out_x0 = self.out_conv(x); out_x1 = self.out_conv1(x);
             out_x2 = self.out_conv2(x); out_x3 = self.out_conv3(x);
             data_format = tf.keras.backend.image_data_format(); channel_axis = -1 if data_format == 'channels_last' else 1;
             x_concat = tf.concat([out_x0, out_x1, out_x2, out_x3], axis=channel_axis);
             data_format_string = "NHWC" if data_format == 'channels_last' else "NCHW";
             x = tf.nn.depth_to_space(x_concat, block_size=2, data_format=data_format_string);
             
             # ---- DEBUG Decoder Logits ----
             tf.print(f"DEBUG Decoder ({self.name}) img_logits min/max/mean:", tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean(x), output_stream=sys.stdout)
             # ------------------------------
             
             x = tf.keras.activations.sigmoid(x);
        else:
             x = self.out_conv(x);
             
             # ---- DEBUG Decoder Logits ----
             tf.print(f"DEBUG Decoder ({self.name}) img_logits min/max/mean:", tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean(x), output_stream=sys.stdout)
             # ------------------------------
             
             x = tf.keras.activations.sigmoid(x);

        # --- Mask Path (NO Checkpointing Logic) ---
        m = z
        num_mask_upscales = len(self.upscalems)
        for i in range(num_mask_upscales):
            # Call layer directly
            m = self.upscalems[i](m)
        # -------------------------------------------

        # --- Final Mask Convolution ---
        m = self.out_convm(m);
        
        # ---- DEBUG Decoder Mask Logits ----
        tf.print(f"DEBUG Decoder ({self.name}) mask_logits min/max/mean:", tf.reduce_min(m), tf.reduce_max(m), tf.reduce_mean(m), output_stream=sys.stdout)
        # ------------------------------
        
        m = tf.keras.activations.sigmoid(m);

        # --- Final Type Casting ---
        if self.use_fp16:
             x = tf.cast(x, tf.float32);
             m = tf.cast(m, tf.float32);
        return x, m

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_ch': self.d_ch,
            'd_mask_ch': self.d_mask_ch,
            'opts': self.opts,
            'activation_name': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation),
            'use_fp16': self.use_fp16,
            'gradient_checkpointing': self.gradient_checkpointing
        })
        return config
    @classmethod
    def from_config(cls, config, custom_objects=None):
         # Requires Conv2D_cls, Upscale_cls, ResidualBlock_cls in custom_objects
         if custom_objects is None or 'Conv2D_cls' not in custom_objects or 'Upscale_cls' not in custom_objects or 'ResidualBlock_cls' not in custom_objects:
              raise ValueError("from_config for Decoder requires Conv2D_cls, Upscale_cls, and ResidualBlock_cls in custom_objects")
         config['Conv2D_cls'] = custom_objects['Conv2D_cls']; config['Upscale_cls'] = custom_objects['Upscale_cls']; config['ResidualBlock_cls'] = custom_objects['ResidualBlock_cls'];
         config.pop('activation_name', None); return cls(**config)

# --- END OF FILE core/leras/archis/Decoder.py ---