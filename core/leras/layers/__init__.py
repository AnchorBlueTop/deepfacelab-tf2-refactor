# --- START OF FILE core/leras/layers/__init__.py --- (Corrected)

# --- REMOVED Saveable Import ---
# from .Saveable import *

# Import the refactored LayerBase (inherits tf.keras.layers.Layer)
from .LayerBase import LayerBase

# Import refactored/verified layers
from .Conv2D import Conv2D
from .Conv2DTranspose import Conv2DTranspose # <-- Assuming this is refactored or TF2 compatible
from .DepthwiseConv2D import DepthwiseConv2D # <-- Assuming this is refactored or TF2 compatible
from .Dense import Dense
from .BlurPool import BlurPool # <-- Needs verification/refactoring

# Import refactored/verified Normalization layers
# from .BatchNorm2D import BatchNorm2D # Needs refactoring (uses running_mean/var)
# from .InstanceNorm2D import InstanceNorm2D # Needs verification/refactoring
# from .FRNorm2D import FRNorm2D # Needs verification/refactoring

# Import refactored/verified Activation/Misc layers
# from .TLU import TLU # Needs verification/refactoring
# from .ScaleAdd import ScaleAdd # Needs verification/refactoring
# from .DenseNorm import DenseNorm # Needs verification/refactoring
# from .AdaIN import AdaIN # Needs verification/refactoring
# from .TanhPolar import TanhPolar # Needs verification (uses nn.bilinear_sampler)

# --- REMOVED MsSsim Import (Moved to losses) ---
# from .MsSsim import *

# Import WScaleConv2D layer
from .WScaleConv2D import WScaleConv2D

from .WScaleDense import WScaleDense

# --- END OF FILE ---