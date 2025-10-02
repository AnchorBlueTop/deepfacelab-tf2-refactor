# --- START OF FILE core/leras/layers/ScaleAdd.py --- (Refactored)

import tensorflow as tf

class ScaleAdd(tf.keras.layers.Layer):
    """
    Adds two inputs, scaling the second input by a learnable weight.
    output = input_1 + input_2 * weight
    """
    def __init__(self, name=None, dtype=None, **kwargs):
        """
        Args:
            name: Layer name.
            dtype: Layer dtype.
            **kwargs: Additional Layer arguments.
        """
        # Note: Channel count 'ch' is inferred in build()
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.scale_weight = None # Weight created in build

    def build(self, input_shape):
        """Create the learnable scale weight based on input channels."""
        # Expects a list/tuple of two input shapes
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("ScaleAdd layer expects a list/tuple of two input shapes.")

        shape1 = tf.TensorShape(input_shape[0])
        shape2 = tf.TensorShape(input_shape[1])

        # Basic compatibility check (rank and spatial dims)
        if shape1.rank != shape2.rank or shape1.rank != 4:
            raise ValueError("Inputs must be 4D tensors.")
        if shape1[1:-1] != shape2[1:-1] and tf.keras.backend.image_data_format() == 'channels_last':
            raise ValueError("Spatial dimensions (H, W) of inputs must match for NHWC.")
        if shape1[2:] != shape2[2:] and tf.keras.backend.image_data_format() == 'channels_first':
             raise ValueError("Spatial dimensions (H, W) of inputs must match for NCHW.")

        # Infer channel count from the second input (the one being scaled)
        data_format = tf.keras.backend.image_data_format()
        channel_axis = -1 if data_format == 'channels_last' else 1
        if shape2[channel_axis] is None:
            raise ValueError("The channel dimension of the second input must be defined.")
        ch = int(shape2[channel_axis])

        layer_dtype = self.dtype or tf.keras.backend.floatx()

        # Create the learnable weight using add_weight
        self.scale_weight = self.add_weight(
            name="scale_weight", # More descriptive than 'weight'
            shape=(ch,),
            initializer=tf.keras.initializers.Zeros(), # Initialize scale factor to 0
            trainable=True,
            dtype=layer_dtype
        )
        super().build(input_shape)

    def call(self, inputs):
        """Perform the scale-and-add operation."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("ScaleAdd layer expects a list/tuple of two inputs: [x0, x1]")
        x0, x1 = inputs

        # Reshape weight for broadcasting
        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_last': # NHWC
            # Shape: (1, 1, 1, ch)
            param_shape = tf.concat( ([1]*(x0.shape.rank-1), [tf.shape(self.scale_weight)[0]]), axis=0 )
        else: # NCHW
            # Shape: (1, ch, 1, 1)
            param_shape = tf.concat( ([1], [tf.shape(self.scale_weight)[0]], [1]*(x0.shape.rank-2)), axis=0 )

        weight_reshaped = tf.reshape(self.scale_weight, param_shape)

        # Perform operation: x0 + x1 * weight
        output = x0 + x1 * weight_reshaped
        return output

    def get_config(self):
        # No specific config needed beyond base Layer args (like name, dtype)
        config = super().get_config()
        return config

# --- REMOVE assignment back ---
# nn.ScaleAdd = ScaleAdd
# --- END OF FILE ---