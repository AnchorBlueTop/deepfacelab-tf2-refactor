# --- START OF FILE core/leras/layers/LayerBase.py --- (Refactored)
import tensorflow as tf
# No longer need to import nn or nn.Saveable here - GOOD

# Inherit directly from the base Keras Layer
class LayerBase(tf.keras.layers.Layer): # CORRECT Inheritance
    """
    Refactored base class for custom leras layers, compatible with Keras API.
    Subclasses should implement build() for weight creation and call() for logic.
    """
    def __init__(self, **kwargs):
        # Ensure Keras Layer __init__ is called
        super().__init__(**kwargs) # CORRECT
        # No need for custom 'built' flag, Keras handles it - GOOD

    # build_weights is replaced by Keras build(self, input_shape) in subclasses - GOOD
    # def build_weights(self): ...

    # forward is replaced by Keras call(self, inputs, **kwargs) in subclasses - GOOD
    # def forward(self, *args, **kwargs): ...

    # __call__ is handled by the parent Keras Layer class - GOOD
    # def __call__(self, *args, **kwargs): ...

# --- REMOVE the line assigning back to nn.LayerBase --- CORRECT
# nn.LayerBase = LayerBase
# --- END OF FILE core/leras/layers/LayerBase.py ---