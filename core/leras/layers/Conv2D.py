# --- START OF FILE core/leras/layers/Conv2D.py ---

import numpy as np
from core.leras import nn
# Ensure the main tensorflow module is imported, not compat.v1
import tensorflow as tf

# Inherit from tf.keras.layers.Layer
class Conv2D(tf.keras.layers.Layer):
    """
    Convolutional 2D Layer compatible with Keras API.

    Supports custom weight scaling (use_wscale).
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, 
                 use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, 
                 trainable=True, dtype=None, name=None, use_fp16 = False, **kwargs ):
        
        # --- Super call ---
        keras_kwargs = { 'name': name, 'dtype': dtype, 'trainable': trainable }
        for k in ['name', 'dtype', 'trainable']: kwargs.pop(k, None)
        if kwargs: print(f"Warning: Conv2D received unexpected kwargs: {kwargs}")
        super().__init__(**keras_kwargs)
        # -----------------
        self._original_in_ch = in_ch # Store for get_config
        self.out_ch = out_ch
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.dilations = int(dilations)
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.use_fp16 = use_fp16
        self.kernel_initializer_arg = kernel_initializer # Store arg for build
        self.bias_initializer_arg = bias_initializer   # Store arg for build

        # Map padding argument to Keras/TF standard strings
        if isinstance(padding, str):
            padding_upper = padding.upper()
            if padding_upper in ('SAME', 'VALID'):
                self.padding = padding_upper
            else:
                # If original code had int padding, it implies custom SAME calculation then VALID conv.
                # Keras 'same' padding handles most cases automatically.
                # Defaulting to SAME if the int logic was for that.
                # If VALID was intended by int=0, the original code had a bug?
                # Assuming non-string padding meant 'SAME' in original intent.
                print(f"Warning: Non-standard padding '{padding}' provided. Interpreting as 'SAME'.")
                self.padding = 'SAME'
        elif isinstance(padding, int):
             # Original code calculated padding value for 'SAME', then did manual pad + 'VALID' conv.
             # Keras 'same' should achieve the equivalent result directly in conv2d.
             print(f"Warning: Integer padding '{padding}' provided. Interpreting as 'SAME'.")
             self.padding = 'SAME'
        else:
             raise ValueError("Padding must be 'SAME', 'VALID', or an integer (interpreted as 'SAME').")


        # Call the parent constructor (tf.keras.layers.Layer)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)


    def build(self, input_shape):
        # Standard Keras method to create weights. Called automatically on first use.
        # Infer input channels from input_shape
        if tf.keras.backend.image_data_format() == "NHWC":
            channel_axis = -1
        else: # NCHW
            channel_axis = 1
        self.in_ch = input_shape[channel_axis]

        if self.in_ch is None:
             raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        # --- Kernel (Weight) Initialization ---
        # Define kernel shape using the inferred self.in_ch
        kernel_shape = (self.kernel_size, self.kernel_size, self.in_ch, self.out_ch)
        kernel_initializer = self.kernel_initializer_arg # Use the stored arg

        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2.0)
            # Calculate fan_in like original code
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = tf.constant(he_std, dtype=self.dtype) # Store wscale factor

            # If using wscale and no initializer provided, use random normal (like original)
            if kernel_initializer is None:
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        else:
            self.wscale = None
            # If not using wscale and no initializer provided, use default Keras initializer (GlorotUniform)
            if kernel_initializer is None:
                 kernel_initializer = tf.keras.initializers.GlorotUniform()


        # Use add_weight() to create the kernel variable
        self.kernel = self.add_weight(
            name='kernel', # Keras uses 'kernel' by convention
            shape=kernel_shape,
            initializer=kernel_initializer,
            trainable=self.trainable,
            dtype=self.dtype)

        # --- Bias Initialization ---
        if self.use_bias:
            bias_initializer = self.bias_initializer_arg # Use the stored arg
            if bias_initializer is None:
                bias_initializer = tf.keras.initializers.Zeros()

            # Use add_weight() to create the bias variable
            self.bias = self.add_weight(
                name='bias',
                shape=(self.out_ch,),
                initializer=bias_initializer,
                trainable=self.trainable,
                dtype=self.dtype)
        else:
            self.bias = None

        # Ensure Keras knows the layer is built
        super().build(input_shape)

    # Replaces 'forward' method
    def call(self, inputs):
        # Apply wscale if enabled
        effective_kernel = self.kernel
        if self.use_wscale and self.wscale is not None:
            effective_kernel = self.kernel * self.wscale

        # Prepare strides and dilations for tf.nn.conv2d format
        # Strides: [batch, height, width, channels] or [batch, channels, height, width]
        # Dilations: [batch, height, width, channels] or [batch, channels, height, width]
        strides_list = [1, self.strides, self.strides, 1]
        dilations_list = [1, self.dilations, self.dilations, 1]
        data_format_string = "NHWC"
        if tf.keras.backend.image_data_format() == "NCHW":
             strides_list = [1, 1, self.strides, self.strides]
             dilations_list = [1, 1, self.dilations, self.dilations]
             data_format_string = "NCHW"

        # Debug print to check shapes at runtime
        from core.interact import interact as io
        io.log_info(f"DEBUG Conv2D.call: Layer name={self.name}, Input shape={tf.shape(inputs)}, Kernel shape={tf.shape(effective_kernel)}")
        tf.print("DEBUG KERNEL SHAPE: ", tf.shape(effective_kernel), ", INPUT SHAPE: ", tf.shape(inputs), ", LAYER: ", self.name)
        
        # Perform convolution using tf.nn.conv2d
        # Keras 'padding' ('SAME' or 'VALID') maps directly to tf.nn.conv2d
        outputs = tf.nn.conv2d(
            inputs,
            effective_kernel,
            strides=strides_list,
            padding=self.padding, # Use 'SAME' or 'VALID'
            dilations=dilations_list,
            data_format=data_format_string,
            name=self.name # Optional name scope
        )

        # Add bias if enabled
        if self.use_bias:
            # Bias add automatically handles data format broadcasting in TF 2.x
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=data_format_string)

        return outputs

    # Optional: Implement get_config for Keras saving/loading compatibility
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_ch': self._original_in_ch, # Add this back
            'out_ch': self.out_ch,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.upper(), # Ensure uppercase string
            'dilations': self.dilations,
            'use_bias': self.use_bias,
            'use_wscale': self.use_wscale,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer_arg),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer_arg),
            # 'trainable' and 'dtype' are handled by the base Layer get_config
        })
        return config


    def __str__(self):
        # Slightly more informative string representation
        return f"{self.__class__.__name__} (out: {self.out_ch}, kernel: {self.kernel_size}, stride: {self.strides}, pad: {self.padding})"

# Assign the refactored class back to nn.Conv2D for compatibility
# nn.Conv2D = Conv2D
# --- END OF FILE core/leras/layers/Conv2D.py ---