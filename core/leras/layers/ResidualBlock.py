# --- START OF FILE core/leras/layers/ResidualBlock.py --- (Corrected Super Call)

import tensorflow as tf

# Inherit from tf.keras.layers.Layer
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, ch, kernel_size=3, activation=None, use_fp16=False, name=None,
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
        if kwargs: print(f"Warning: ResidualBlock received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------------------------------------------------

        # Now process custom arguments
        self.ch = ch
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.use_fp16 = use_fp16
        self.conv_dtype = tf.float16 if use_fp16 else (self.dtype or tf.keras.backend.floatx())

        # --- Store Conv2D class reference ---
        if Conv2D_cls is None or not issubclass(Conv2D_cls, tf.keras.layers.Layer):
             raise ValueError("ResidualBlock requires a valid Conv2D_cls (tf.keras.layers.Layer).")
        self.Conv2D_cls = Conv2D_cls
        # -----------------------------------

        # --- FORCE using standard Keras Conv2D parameters ---
        self.conv1 = self.Conv2D_cls(filters=self.ch, kernel_size=self.kernel_size, padding='SAME', name=f"{self.name}_conv1" if self.name else None)
        self.conv2 = self.Conv2D_cls(filters=self.ch, kernel_size=self.kernel_size, padding='SAME', name=f"{self.name}_conv2" if self.name else None)
        self.activation_func = tf.keras.activations.get(activation)
        print(f"DEBUG ResidualBlock: Built Conv2D with class {self.Conv2D_cls.__name__}")
        # ---------------------------------
    # --- End of __init__ ---

    def call(self, inputs, training=False): # Add training arg if needed by activation/other logic
        x = self.conv1(inputs);
        if self.activation_func is not None: # Apply stored activation
            # Original used alpha=0.2 for leaky relu - handle potential args if needed
            try: x = self.activation_func(x, alpha=0.2)
            except TypeError: x = self.activation_func(x)

        x = self.conv2(x);
        x = inputs + x; # Residual connection

        if self.activation_func is not None: # Apply stored activation again
            try: x = self.activation_func(x, alpha=0.2)
            except TypeError: x = self.activation_func(x)
        return x

    def get_config(self):
        config = super().get_config(); config.update({'ch': self.ch, 'kernel_size': self.kernel_size, 'activation_name': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation), 'use_fp16': self.use_fp16}); return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Requires Conv2D_cls in custom_objects
         if custom_objects is None or 'Conv2D_cls' not in custom_objects:
              raise ValueError("from_config for ResidualBlock requires Conv2D_cls in custom_objects")
         config['Conv2D_cls'] = custom_objects['Conv2D_cls']
         config.pop('activation_name', None); return cls(**config)

    def __str__(self): return f"{self.__class__.__name__} (ch: {self.ch}, kernel: {self.kernel_size})"

# --- END OF FILE core/leras/layers/ResidualBlock.py ---