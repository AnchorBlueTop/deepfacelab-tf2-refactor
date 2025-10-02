# At top of core/leras/archis/DeepFakeArchi.py
import tensorflow as tf
from .ArchiBase import ArchiBase # Assuming this exists and is simple
try:
    from .Encoder import Encoder
    print(f"DEBUG DeepFakeArchi: Imported Encoder type: {type(Encoder)}")
except ImportError as e: print(f"ERROR DeepFakeArchi: Cannot import Encoder: {e}"); Encoder = object
try:
    from .Inter import Inter
    print(f"DEBUG DeepFakeArchi: Imported Inter type: {type(Inter)}")
except ImportError as e: print(f"ERROR DeepFakeArchi: Cannot import Inter: {e}"); Inter = object
try:
    from .Decoder import Decoder
    print(f"DEBUG DeepFakeArchi: Imported Decoder type: {type(Decoder)}")
except ImportError as e: print(f"ERROR DeepFakeArchi: Cannot import Decoder: {e}"); Decoder = object

class DeepFakeArchi(tf.keras.Model):
    """
    Refactored DeepFakeArchi.
    Acts as a namespace/factory returning the Keras-compatible
    Encoder, Inter, and Decoder classes/models.
    The actual layer definitions are now in separate files.
    """
    def __init__(self, resolution, use_fp16=False, mod=None, opts=None):
        super().__init__() # Call ArchiBase init if it has one

        # Store config potentially needed by components, though ideally components
        # get config passed directly during instantiation in SAEHDModel.
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.opts = opts if opts is not None else ''
        self.mod = mod # Store mod ('quick', 'uhd', None) if needed

        # --- Assign the imported classes as attributes ---
        # This allows SAEHDModel to access them via model_archi.Encoder etc.

        # Note: The original code had different structures based on 'mod'.
        # We need to replicate that structure selection if mods ('quick', 'uhd')
        # actually used *different* Encoder/Inter/Decoder implementations.
        # If they only changed parameters (like channels, kernel sizes), then
        # SAEHDModel handles that parameter passing during instantiation.
        # Assuming for now that the components themselves are the same structure
        # regardless of mod, and only parameters change.

        # TODO: Verify if 'quick' or 'uhd' mods require fundamentally different
        # Encoder/Inter/Decoder *classes* vs just different parameters passed to them.
        # If they need different classes, you'd need conditional imports/assignments here.

        self.Encoder = Encoder # Assign the imported refactored class
        self.Inter = Inter     # Assign the imported refactored class
        self.Decoder = Decoder # Assign the imported refactored class

        # These component classes (Upscale, etc.) are usually instantiated directly
        # within Encoder/Inter/Decoder, so no need to expose them here unless
        # SAEHDModel itself was instantiating them directly.
        # self.Upscale = Upscale
        # self.ResidualBlock = ResidualBlock
        # self.Downscale = Downscale
        # self.DownscaleBlock = DownscaleBlock


    # Remove old methods that returned nested classes or had TF1 logic
    # def get_model_filename_list(self): ...
    # def flow(self, ...): ... # ArchiBase already raises NotImplementedError


# Assign back to nn namespace
# nn.DeepFakeArchi = DeepFakeArchi
# --- END OF FILE core/leras/archis/DeepFakeArchi.py ---