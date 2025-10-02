# --- START OF FILE core/leras/layers/Upscale.py --- (Corrected Super Call)

import tensorflow as tf
from core.leras import nn  # Import for data_format

# Inherit from tf.keras.layers.Layer
class Upscale(tf.keras.layers.Layer):
    def __init__(self, out_ch, kernel_size=3, activation=None, use_fp16=False, name=None,
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
        if kwargs: print(f"Warning: Upscale received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------------------------------------------------

        # Now process custom arguments
        self.out_ch = out_ch
        self.kernel_size = int(kernel_size)
        self.activation = tf.keras.activations.get(activation)
        self.use_fp16 = use_fp16 # Keep for get_config maybe, though mixed precision handles it mostly
        self.conv_dtype = tf.float16 if use_fp16 else (self.dtype or tf.keras.backend.floatx())

        # --- Dependency Injection Fallback ---
        if Conv2D_cls is None:
            # Attempt to import standard Keras layer as fallback
            try:
                from tensorflow.keras.layers import Conv2D as Conv2D_cls
                print(f"Upscale '{name}': Conv2D_cls not provided, using fallback tf.keras.layers.Conv2D.")
            except ImportError:
                # Or fallback to custom if that's intended? Requires careful check.
                # For now, assume standard Keras is the target if None is passed.
                raise ImportError("Upscale layer requires Conv2D_cls argument or tensorflow.keras.layers to be available.")
        self.Conv2D_cls = Conv2D_cls # Store the class
        # ------------------------------------
        self.conv1 = None # Defined in build

    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        channel_axis = -1 if data_format == 'channels_last' else 1
        in_ch = input_shape[channel_axis]
        if in_ch is None: raise ValueError('Input channel dimension must be defined.')

        # --- Use the stored Conv2D_cls ---
        if nn.data_format == "NHWC": 
            channel_axis = -1
        else: 
            channel_axis = 1
        in_ch = input_shape[channel_axis];
        if in_ch is None: raise ValueError('Input channel dimension must be defined.')
        self.conv_dtype = tf.float16 if self.use_fp16 else tf.float32 # Define dtype
        # FORCE use of standard Keras Conv2D args
        # This should already be using filters instead of out_ch/in_ch
        self.conv1 = self.Conv2D_cls(filters=self.out_ch * 4, kernel_size=self.kernel_size, padding='SAME', name=f"{self.name}_conv2d" if self.name else None)
        print(f"DEBUG Upscale: Built Conv2D with class {self.Conv2D_cls.__name__}")
        # ---------------------------------
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)

        if self.activation is not None:
            x = self.activation(x)

        data_format_keras = tf.keras.backend.image_data_format()
        data_format_string = "NHWC" if data_format_keras == 'channels_last' else "NCHW"
        outputs = tf.nn.depth_to_space(x, block_size=2, data_format=data_format_string)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_ch': self.out_ch,
            'kernel_size': self.kernel_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_fp16': self.use_fp16,
            # Cannot serialize Conv2D_cls easily
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
         # Requires Conv2D_cls in custom_objects
         if custom_objects is None or 'Conv2D_cls' not in custom_objects:
              raise ValueError("from_config for Upscale requires Conv2D_cls in custom_objects")
         config['Conv2D_cls'] = custom_objects['Conv2D_cls']
         config['activation'] = tf.keras.activations.deserialize(config.get('activation'))
         return cls(**config)

    def __str__(self):
        return f"{self.__class__.__name__} (out_ch: {self.out_ch}, kernel: {self.kernel_size})"

# --- END OF FILE core/leras/layers/Upscale.py ---