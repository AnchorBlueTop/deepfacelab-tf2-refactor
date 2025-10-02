# --- START OF FILE core/leras/layers/Downscale.py --- (Corrected Super Call)

import tensorflow as tf
from core.leras import nn  # Import for data_format

class Downscale(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, kernel_size=5, activation=None, use_fp16=False, name=None,
                 # --- Accept Conv2D class ---
                 Conv2D_cls=None,
                 **kwargs):

        # --- Call super() FIRST, passing ONLY known Keras args ---
        keras_kwargs = {
            'name': name,
            'dtype': kwargs.get('dtype', None)
        }
        kwargs.pop('name', None)
        kwargs.pop('dtype', None)
        if kwargs: print(f"Warning: Downscale received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------------------------------------------------

        # Now process custom arguments
        self.in_ch_original = in_ch # Store original in_ch
        self.out_ch = out_ch
        self.kernel_size = int(kernel_size)
        self.activation = tf.keras.activations.get(activation)
        self.use_fp16 = use_fp16
        self.conv_dtype = tf.float16 if use_fp16 else (self.dtype or tf.keras.backend.floatx())

        # --- Store Conv2D class reference ---
        if Conv2D_cls is None or not issubclass(Conv2D_cls, tf.keras.layers.Layer):
             raise ValueError("Downscale requires a valid Conv2D_cls (tf.keras.layers.Layer).")
        self.Conv2D_cls = Conv2D_cls
        # -----------------------------------

        # --- FORCE using standard Keras Conv2D parameters ---
        self.conv1 = self.Conv2D_cls(filters=self.out_ch, kernel_size=self.kernel_size,
                           strides=(2, 2), padding='SAME',
                           activation=self.activation,
                           name=f"{self.name}_conv2d" if self.name else None)
        print(f"DEBUG Downscale: Built Conv2D with class {self.Conv2D_cls.__name__}")
        # ---------------------------------
    # --- End of __init__ ---

    def call(self, inputs):
        x = self.conv1(inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_ch': self.in_ch_original, # Add this back
            'out_ch': self.out_ch, 
            'kernel_size': self.kernel_size, 
            'activation': tf.keras.activations.serialize(self.activation), 
            'use_fp16': self.use_fp16,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Requires Conv2D_cls in custom_objects
         if custom_objects is None or 'Conv2D_cls' not in custom_objects:
              raise ValueError("from_config for Downscale requires Conv2D_cls in custom_objects")
         config['Conv2D_cls'] = custom_objects['Conv2D_cls']
         config['activation'] = tf.keras.activations.deserialize(config.get('activation'))
         return cls(**config)

    def __str__(self): return f"{self.__class__.__name__} (out_ch: {self.out_ch}, kernel: {self.kernel_size})"

# --- END OF FILE core/leras/layers/Downscale.py ---