# --- START OF FILE core/leras/layers/DownscaleBlock.py --- (Corrected Super Call)

import tensorflow as tf
from core.interact import interact as io

class DownscaleBlock(tf.keras.layers.Layer):
    def __init__(self, input_ch=None, in_ch=None, ch=64, n_downscales=4, kernel_size=3, activation=None, use_fp16=False, name=None,
                 # --- Accept Downscale and Conv2D classes ---
                 Downscale_cls=None,
                 Conv2D_cls=None,
                 **kwargs):

        # --- Call super() FIRST, passing ONLY known Keras args ---
        keras_kwargs = {
            'name': name,
            'dtype': kwargs.get('dtype', None)
        }
        kwargs.pop('name', None)
        kwargs.pop('dtype', None)
        if kwargs: print(f"Warning: DownscaleBlock received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------------------------------------------------

        # Now process custom arguments
        self.ch = ch
        self.n_downscales = n_downscales
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.use_fp16 = use_fp16

        # --- Determine initial input channel count ---
        # Priority: input_ch > in_ch > default
        if input_ch is not None:
            self.in_ch_initial = input_ch
            io.log_info(f"DownscaleBlock '{name}': Using provided input_ch={input_ch} for the first downscale layer.")
        elif in_ch is not None:
            self.in_ch_initial = in_ch
            io.log_info(f"DownscaleBlock '{name}': Using provided in_ch={in_ch} for the first downscale layer.")
        else:
            self.in_ch_initial = ch  # Default to base channel count
            io.log_info(f"DownscaleBlock '{name}': No input_ch or in_ch provided. Using ch={ch} for the first downscale layer.")
        # -----------------------------------------------

        # --- Store class references ---
        if Downscale_cls is None or not issubclass(Downscale_cls, tf.keras.layers.Layer):
             raise ValueError("DownscaleBlock requires a valid Downscale_cls.")
        if Conv2D_cls is None or not issubclass(Conv2D_cls, tf.keras.layers.Layer):
             raise ValueError("DownscaleBlock requires a valid Conv2D_cls.")
        self.Downscale_cls = Downscale_cls
        self.Conv2D_cls = Conv2D_cls
        # -----------------------------

        self.down_layers = []
        # Corrected loop and final conv instantiation
        current_in_ch = input_ch if input_ch is not None else ch # Initialize starting input dimension

        for i in range(self.n_downscales):
            layer_in_ch = current_in_ch
            # Output channels double at each step relative to the *base* channel 'ch'
            layer_out_ch = ch * (2**i) * 2 # Calculate output channels: ch*2, ch*4, ch*8, ch*16...

            io.log_info(f"DownscaleBlock '{name}' layer {i}: Creating Downscale with in_ch={layer_in_ch}, out_ch={layer_out_ch}")
            self.down_layers.append(
                Downscale_cls(in_ch=layer_in_ch, out_ch=layer_out_ch, kernel_size=self.kernel_size,
                              activation=self.activation, use_fp16=self.use_fp16,
                              name=f"{name}_down_{i}" if name else None,
                              Conv2D_cls=Conv2D_cls)
            )
            # Input for the next layer is the output of this one
            current_in_ch = layer_out_ch

        # Final convolution layer to potentially adjust channels before flattening
        # Its input channels are the output channels of the last downscale layer
        final_conv_in_ch = current_in_ch
        final_conv_out_ch = final_conv_in_ch
        io.log_info(f"DownscaleBlock '{name}' final_conv: Creating Conv2D with filters={final_conv_out_ch}")
        self.final_conv = self.Conv2D_cls(filters=final_conv_out_ch, kernel_size=3, padding='SAME', name=f"{name}_final_conv" if name else None)
        self.final_act = tf.keras.layers.LeakyReLU(alpha=0.2)
        print(f"DEBUG DownscaleBlock: Built Conv2D with class {self.Conv2D_cls.__name__}")
        # self.final_act is defined after this block in the original __init__

    def call(self, inputs, training=False): # Keep training arg if needed elsewhere
        x = inputs
        for down_layer in self.down_layers:
            # Standard Downscale.call doesn't need training passed explicitly here
            x = down_layer(x)
        # Standard Conv2D doesn't need training passed explicitly here
        x = self.final_conv(x)
        x = self.final_act(x) # Apply activation after conv
        return x

    def get_config(self):
         config = super().get_config()
         config.update({
             'input_ch': self.in_ch_initial,  # Store as input_ch for consistency
             'in_ch': None,  # Keep for backward compatibility
             'ch': self.ch,
             'n_downscales': self.n_downscales,
             'kernel_size': self.kernel_size,
             'activation': tf.keras.activations.serialize(self.activation),
             'use_fp16': self.use_fp16,
         })
         return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
         # Requires Downscale_cls and Conv2D_cls in custom_objects
         if custom_objects is None or 'Downscale_cls' not in custom_objects or 'Conv2D_cls' not in custom_objects:
              raise ValueError("from_config for DownscaleBlock requires Downscale_cls and Conv2D_cls in custom_objects")
         config['Downscale_cls'] = custom_objects['Downscale_cls']
         config['Conv2D_cls'] = custom_objects['Conv2D_cls']
         config['activation'] = tf.keras.activations.deserialize(config.get('activation'))
         return cls(**config)

    def __str__(self): return f"{self.__class__.__name__} (n_downscales: {self.n_downscales}, base_ch: {self.ch})"

# --- END OF FILE core/leras/layers/DownscaleBlock.py ---