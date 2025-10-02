# --- START OF FILE models/Model_SAEHD/Model.py --- (Part 1/3)

# Python Standard Libraries
import multiprocessing
import operator
import time
import datetime
import pickle
import shutil
import traceback
import colorsys
import sys
from pathlib import Path
import copy
try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML is not installed. Config file saving/loading will be limited. Install with 'pip install PyYAML'")

# Third-party Libraries
import numpy as np
import tensorflow as tf
import cv2

# DFL Core Libraries
from core import mathlib, pathex, imagelib
from core.interact import interact as io
from core.leras import nn # Needed for global config like data_format

# --- Leras Architecture Components ---
try:
    from core.leras.archis.Encoder import Encoder
    from core.leras.archis.Inter import Inter
    from core.leras.archis.Decoder import Decoder
    print("DEBUG: Imported Archi Components directly from modules.")
except ImportError as e:
    io.log_err(f"FATAL: Cannot import archis - {e}")
    raise e

# --- Leras Layer Components ---
try:
    # Import WScale layers
    from core.leras.layers.WScaleConv2D import WScaleConv2D
    from core.leras.layers.WScaleDense import WScaleDense
    print("DEBUG: Imported WScaleConv2D and WScaleDense.")
    
    # Keep other necessary layer imports
    from core.leras.layers.Dense import Dense
    from core.leras.layers.Upscale import Upscale
    from core.leras.layers.ResidualBlock import ResidualBlock
    from core.leras.layers.Downscale import Downscale
    from core.leras.layers.DownscaleBlock import DownscaleBlock
    print("DEBUG: Imported Other Layer Classes directly from modules.")
except ImportError as e:
    io.log_err(f"FATAL: Cannot import required layer classes - {e}")
    raise e

# --- Leras Optimizer(s) ---
try:
    # Attempt to import refactored RMSprop first
    from core.leras.optimizers import RMSprop
    print("DEBUG: Imported refactored RMSprop Optimizer.")
except ImportError:
    io.log_info("Warning: Cannot import refactored leras.RMSprop. Using Keras default.")
    # Fallback to Keras RMSprop
    from tensorflow.keras.optimizers import RMSprop as KerasRMSprop
    # Alias it if needed, or use KerasRMSprop directly later
    RMSprop = KerasRMSprop

# --- Leras Loss Function(s) ---
DssimLoss_class = None
MsSsimLoss_class = None # Use single 'S' for the variable name as well
try:
    from core.leras.losses.DssimLoss import DssimLoss as DssimLoss_imported
    DssimLoss_class = DssimLoss_imported
    print(f"DEBUG: Successfully imported DssimLoss: {type(DssimLoss_class)}")
except ImportError as e_dssim:
    io.log_err(f"Warning: Cannot import DssimLoss directly - {e_dssim}")
    DssimLoss_class = None
except Exception as e_dssim_other:
    io.log_err(f"Warning: Other error importing DssimLoss directly - {e_dssim_other}")
    DssimLoss_class = None
    
try:
    from core.leras.losses.MsSsimLoss import MsSsimLoss as MsSsimLoss_imported # Use single 'S' for filename
    MsSsimLoss_class = MsSsimLoss_imported
    print(f"DEBUG: Successfully imported MsSsimLoss: {type(MsSsimLoss_class)}") # Use single 'S'
except ImportError as e_mssim:
    io.log_err(f"Warning: Cannot import MsSsimLoss directly - {e_mssim}") # Use single 'S'
    MsSsimLoss_class = None
except Exception as e_mssim_other:
    io.log_err(f"Warning: Other error importing MsSsimLoss directly - {e_mssim_other}") # Use single 'S'
    MsSsimLoss_class = None
# Assign to class attributes or local variables as needed by the rest of the class
# For now, let's keep them as DssimLoss and MsSsimLoss to match later usage,
# but ensure they are assigned the potentially None classes.
DssimLoss = DssimLoss_class
MsSsimLoss = MsSsimLoss_class # Use single 'S'
print(f"DEBUG Final Check: DssimLoss is {type(DssimLoss)}, MsSsimLoss is {type(MsSsimLoss)}") # Use single 'S'
# ----------------------------

# --- Leras Discriminator Models (Optional) ---
try:
    # Assume these might still be TF1 based or need refactoring
    # from core.leras.models.CodeDiscriminator import CodeDiscriminator
    # from core.leras.models.PatchDiscriminator import UNetPatchDiscriminator
    CodeDiscriminator = None # Temporarily disable until refactored
    UNetPatchDiscriminator = None # Temporarily disable until refactored
    print("DEBUG: Discriminator Models Temporarily Disabled.")
except ImportError:
    io.log_info("Warning: Cannot import discriminators (CodeDiscriminator or UNetPatchDiscriminator).")
    CodeDiscriminator=UNetPatchDiscriminator=None # Define as None

# --- Import Top-Level Packages/Modules ONLY ---
try:
    import samplelib # Import the package
    from samplelib import SampleProcessor # Import the specific class
    import facelib   # Import the package
    print("DEBUG: Imported samplelib and facelib packages.")
except ImportError as e:
    io.log_err(f"FATAL: Cannot import samplelib or facelib package - {e}")
    raise e

# --- Utility Components ---
try:
    from utils.label_face import label_face_filename
except ImportError as e:
     io.log_err(f"Warning: Cannot import label_face_filename - {e}")
     def label_face_filename(img, text): return img # Dummy fallback

# --- End of Import Section ---


# --- Class Definition ---
class Model(tf.keras.Model): # Renamed back to Model

    def __init__(self,
                 is_training=False,
                 saved_models_path=None,
                 training_data_src_path=None,
                 training_data_dst_path=None,
                 pretraining_data_path=None,
                 no_preview=False,
                 force_model_name=None,
                 silent_start=False,
                 config_training_file=None,
                 auto_gen_config=False,
                 force_gradient_checkpointing=False,
                 debug=False,
                 batch_size=4, # Default, might be overridden by options
                 **kwargs):

        # Determine Model Name FIRST
        # Default to parent directory name if not forced
        default_model_name = Path(__file__).parent.name
        determined_model_name = force_model_name if force_model_name is not None else default_model_name

        # Call the parent Keras Model __init__ EARLY
        super().__init__(name=determined_model_name, **kwargs)
        io.log_info(f"Initializing {self.name} model.")

        # Store essential flags and paths
        self.is_training = is_training
        self.is_exporting = kwargs.get('is_exporting', False)
        self.saved_models_path = Path(saved_models_path) if saved_models_path is not None else None
        self.training_data_src_path = Path(training_data_src_path) if training_data_src_path is not None else None
        self.training_data_dst_path = Path(training_data_dst_path) if training_data_dst_path is not None else None
        self.pretraining_data_path = Path(pretraining_data_path) if pretraining_data_path is not None else None
        self.no_preview = no_preview
        self.silent_start = silent_start # Store silent_start flag
        self.debug = debug
        self.batch_size = batch_size # Store initial batch size

        # Load existing data AFTER super().__init__
        self._iter_from_save = 0
        loaded_options = None
        loaded_loss_history = []
        loaded_sample_for_preview = None
        # Use helper method to get path
        self.model_data_path = self.get_strpath_storage_for_file('data.dat')

        if self.model_data_path is None:
             io.log_err("Could not determine model data path! Saving may fail.")
             # Attempt to create a default path? Risky.
             # self.model_data_path = Path(f'./{self.name}_data.dat')
        elif self.model_data_path.exists():
            io.log_info (f"Loading existing model data from {self.model_data_path}...")
            try:
                model_data = pickle.loads( self.model_data_path.read_bytes() )
                self._iter_from_save = model_data.get('iter',0)
                loaded_options = model_data.get('options', {})
                loaded_loss_history = model_data.get('loss_history', [])
                loaded_sample_for_preview = model_data.get('sample_for_preview', None)
                io.log_info(f"Loaded iter: {self._iter_from_save}")
            except Exception as e:
                io.log_err(f"Could not load model data from {self.model_data_path}: {e}"); loaded_options = {}
        else:
             io.log_info(f"Model data file not found: {self.model_data_path}. Initializing fresh options."); loaded_options = {}

        self.options = loaded_options if loaded_options is not None else {}
        self.loss_history = loaded_loss_history

        # Option Loading and Processing
        self.options_show_override = {}
        self.read_from_conf = config_training_file is not None
        self.config_training_file = Path(config_training_file) if config_training_file is not None else None
        self.config_file_exists = self.config_training_file is not None and self.config_training_file.exists()
        self.auto_gen_config = auto_gen_config
        self.initialize_options(silent_start, force_gradient_checkpointing, batch_size) # Pass silent_start

        # Initialize Model Architecture (using final options)
        self.resolution = self.options['resolution']
        io.log_info(f"DEBUG Model __init__: self.resolution (from options): {self.resolution}") # Added
        self.ae_dims = self.options['ae_dims']
        self.e_dims = self.options['e_dims']
        self.d_dims = self.options['d_dims']
        self.d_mask_dims = self.options['d_mask_dims']
        self.archi = self.options['archi']
        self.archi_type, self.archi_opts = self._parse_archi()
        self.use_fp16 = self.options['use_fp16']
        self.face_type_str = self.options.get('face_type', 'full_face')
        try: self.face_type_enum = facelib.FaceType.fromString(self.face_type_str)
        except Exception as e: io.log_err(f"Error converting face_type '{self.face_type_str}' to enum: {e}. Using default FULL_FACE."); self.face_type_enum = facelib.FaceType.FULL # Fallback

        # Set backend floatx - Note: This affects ALL layers.
        # Consider using mixed precision API for more control if needed later.
        if self.use_fp16:
            tf.keras.backend.set_floatx('float16')
            self.conv_dtype = tf.float16
            io.log_info("Using float16 precision.")
        else:
            tf.keras.backend.set_floatx('float32')
            self.conv_dtype = tf.float32
            io.log_info("Using float32 precision.")

        # Define activation function
        if 'c' in self.archi_opts: self.activation_func = lambda x: x * tf.cos(x)
        else: self.activation_func = tf.keras.layers.LeakyReLU(alpha=0.1) # Use Keras LeakyReLU layer

        # Calculate lowest_dense_res
        lowest_dense_res = self.resolution // (32 if 'd' in self.archi_opts else 16)

        # Instantiate Architecture Components
        input_ch = 3
        io.log_info(f"Architecture type: {self.archi_type}, Options: {self.archi_opts}")

        # Get base classes to pass down to layers
        BaseDense_cls = WScaleDense
        BaseConv2D_cls = WScaleConv2D
        io.log_info(f"Using WScaleConv2D (type: {type(BaseConv2D_cls)}) and WScaleDense (type: {type(BaseDense_cls)}) for architectures.")

        # Create ARCHITECTURE instances, passing necessary layer classes
        # Ensure the layer classes (Upscale, ResidualBlock etc.) are correctly imported
        # Also pass the gradient_checkpointing option
        gradient_checkpointing = self.options.get('gradient_checkpointing', False)
        
        if 'df' in self.archi_type:
             self.encoder = Encoder(e_ch=self.e_dims, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='encoder', 
                                   Downscale_cls=Downscale, DownscaleBlock_cls=DownscaleBlock, ResidualBlock_cls=ResidualBlock, Conv2D_cls=BaseConv2D_cls,
                                   gradient_checkpointing=gradient_checkpointing)
             self.inter = Inter(ae_ch=self.ae_dims, ae_out_ch=self.ae_dims, lowest_dense_res=lowest_dense_res, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='inter', Dense_cls=BaseDense_cls, Upscale_cls=Upscale, Conv2D_cls=BaseConv2D_cls)
             self.decoder_src = Decoder(d_ch=self.d_dims, d_mask_ch=self.d_mask_dims, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='decoder_src', Conv2D_cls=BaseConv2D_cls, Upscale_cls=Upscale, ResidualBlock_cls=ResidualBlock, gradient_checkpointing=gradient_checkpointing)
             self.decoder_dst = Decoder(d_ch=self.d_dims, d_mask_ch=self.d_mask_dims, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='decoder_dst', Conv2D_cls=BaseConv2D_cls, Upscale_cls=Upscale, ResidualBlock_cls=ResidualBlock, gradient_checkpointing=gradient_checkpointing)
        elif 'liae' in self.archi_type:
             self.encoder = Encoder(e_ch=self.e_dims, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='encoder', 
                                   Downscale_cls=Downscale, DownscaleBlock_cls=DownscaleBlock, ResidualBlock_cls=ResidualBlock, Conv2D_cls=BaseConv2D_cls,
                                   gradient_checkpointing=gradient_checkpointing)
             self.inter_AB = Inter(ae_ch=self.ae_dims, ae_out_ch=self.ae_dims*2, lowest_dense_res=lowest_dense_res, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='inter_AB', Dense_cls=BaseDense_cls, Upscale_cls=Upscale, Conv2D_cls=BaseConv2D_cls)
             self.inter_B  = Inter(ae_ch=self.ae_dims, ae_out_ch=self.ae_dims*2, lowest_dense_res=lowest_dense_res, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='inter_B', Dense_cls=BaseDense_cls, Upscale_cls=Upscale, Conv2D_cls=BaseConv2D_cls)
             self.decoder  = Decoder(d_ch=self.d_dims, d_mask_ch=self.d_mask_dims, opts=self.archi_opts, activation=self.activation_func, use_fp16=self.use_fp16, name='decoder', Conv2D_cls=BaseConv2D_cls, Upscale_cls=Upscale, ResidualBlock_cls=ResidualBlock, gradient_checkpointing=gradient_checkpointing)
        else: raise ValueError(f"Unknown archi_type: {self.archi_type}")

        # Define Discriminators (Currently disabled)
        self.code_discriminator = None; self.D_src = None
        # if self.is_training and self.options.get('true_face_power', 0.0) > 0.0 and 'df' in self.archi_type: ...
        # if self.is_training and self.options.get('gan_power', 0.0) > 0.0: ...

        # Define Optimizers
        self.optimizer_G = None; self.optimizer_D_code = None; self.optimizer_D_gan = None
        if self.is_training:
            lr_opt = self.options.get('lr', 5e-5)
            lr_opt = tf.cast(lr_opt, tf.float32) # Explicitly cast LR to float32
            clipvalue_opt = 1.0 if self.options.get('clipgrad', False) else None
            self.lr_dropout_rate = 0.3 if self.options.get('lr_dropout', 'n') in ['y', 'cpu'] else 1.0
            lr_schedule = lr_opt; lr_cos_steps = self.options.get('lr_cos', 0)
            if lr_cos_steps > 0:
                io.log_info(f"Using Cosine Decay for LR: {lr_cos_steps} steps.")
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr_opt, max(1, lr_cos_steps), alpha=0.0)

            # Try using refactored RMSprop (should be imported correctly now)
            OptimizerClass = RMSprop # Should be the custom one
            optimizer_kwargs = {
                'learning_rate': lr_schedule,
                'rho': 0.9,
                'epsilon': 1e-7,
                'clipvalue': clipvalue_opt,
                'name': "RMSprop_G"
            }
            # Add lr_dropout_rate only if using the custom optimizer
            if 'lr_dropout_rate' in OptimizerClass.__init__.__code__.co_varnames: # Check if custom optimizer supports it
                 optimizer_kwargs['lr_dropout_rate'] = self.lr_dropout_rate
                 io.log_info(f"Using custom leras.RMSprop (LRDrop Rate: {self.lr_dropout_rate}).")
            else:
                 io.log_info(f"Using standard Keras Optimizer (RMSprop fallback).") # Should not happen if custom RMSprop imports

            self.optimizer_G = OptimizerClass(**optimizer_kwargs)

            # Discriminator optimizers (currently discriminators disabled)
            # if self.code_discriminator is not None: self.optimizer_D_code = tf.keras.optimizers.Adam(...)
            # if self.D_src is not None: self.optimizer_D_gan = tf.keras.optimizers.Adam(...)

            if self._iter_from_save > 0:
                # Need to set iteration AFTER optimizer is created
                self.set_iter(self._iter_from_save)
                io.log_info(f"Set optimizer iterations to {self._iter_from_save}")

        # Initialize Sample Generators
        self.training_data_src = None; self.training_data_dst = None; self.generator_list = []
        if self.is_training: self.initialize_sample_generators()

        # Load sample_for_preview
        self.sample_for_preview = loaded_sample_for_preview
        if self.sample_for_preview is None and self.is_training: self.update_sample_for_preview(force_new=True)

        io.log_info(f"{self.name} initialized.")

    def initialize_options(self, silent_start, force_gradient_checkpointing, default_batch_size):
        """Loads default/saved options, potentially asks user interactively, and finalizes them."""
        io.log_info("Initializing options...")

        # --- Load defaults/saved values for ALL options FIRST ---
        # Use self.load_or_def_option helper
        # Core Arch Params
        self.options['resolution'] = self.load_or_def_option('resolution', 256)
        self.options['face_type'] = self.load_or_def_option('face_type', 'whole_face')
        self.options['archi'] = self.load_or_def_option('archi', 'liae-udt')
        self.options['ae_dims'] = self.load_or_def_option('ae_dims', 292)
        self.options['e_dims'] = self.load_or_def_option('e_dims', 78)
        self.options['d_dims'] = self.load_or_def_option('d_dims', 78)
        # Use user-specified default for d_mask_dims
        self.options['d_mask_dims'] = self.load_or_def_option('d_mask_dims', 32)

        # Training Features
        self.options['masked_training'] = self.load_or_def_option('masked_training', True)
        self.options['eyes_mouth_prio'] = self.load_or_def_option('eyes_mouth_prio', False) # Renamed from eyes_prio / mouth_prio for simplicity
        self.options['uniform_yaw'] = self.load_or_def_option('uniform_yaw', False)
        self.options['blur_out_mask'] = self.load_or_def_option('blur_out_mask', False)
        self.options['place_models_on_gpu'] = self.load_or_def_option('models_opt_on_gpu', True) # Use original key

        # Optimizer Params
        self.options['adabelief'] = self.load_or_def_option('adabelief', False)
        lr_dropout_opt = self.load_or_def_option('lr_dropout', 'n')
        self.options['lr_dropout'] = {True:'y', False:'n'}.get(lr_dropout_opt, lr_dropout_opt)
        self.options['lr'] = self.load_or_def_option('lr', 5e-5)
        self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)
        self.options['lr_cos'] = self.load_or_def_option('lr_cos', 0)

        # Loss Params
        self.options['loss_function'] = self.load_or_def_option('loss_function', 'SSIM')
        self.options['true_face_power'] = self.load_or_def_option('true_face_power', 0.0)
        self.options['face_style_power'] = self.load_or_def_option('face_style_power', 0.0)
        self.options['bg_style_power'] = self.load_or_def_option('bg_style_power', 0.0)
        self.options['background_power'] = self.load_or_def_option('background_power', 0.0) # Added

        # GAN Params
        self.options['gan_power'] = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size = self.options.get('resolution', 256) // 8 # Use current default res
        self.options['gan_patch_size'] = self.load_or_def_option('gan_patch_size', default_gan_patch_size)
        self.options['gan_dims'] = self.load_or_def_option('gan_dims', 16)
        self.options['gan_smoothing'] = self.load_or_def_option('gan_smoothing', 0.0)
        self.options['gan_noise'] = self.load_or_def_option('gan_noise', 0.0)

        # Data Augmentation Params
        self.options['random_warp'] = self.load_or_def_option('random_warp', True)
        self.options['random_hsv_power'] = self.load_or_def_option('random_hsv_power', 0.0)
        self.options['random_downsample'] = self.load_or_def_option('random_downsample', False)
        self.options['random_noise'] = self.load_or_def_option('random_noise', False)
        self.options['random_blur'] = self.load_or_def_option('random_blur', False)
        self.options['random_jpeg'] = self.load_or_def_option('random_jpeg', False)
        self.options['random_shadow'] = self.load_or_def_option('random_shadow', 'none')
        self.options['ct_mode'] = self.load_or_def_option('ct_mode', 'none')
        self.options['random_color'] = self.load_or_def_option('random_color', False)
        self.options['random_src_flip'] = self.load_or_def_option('random_src_flip', False)
        self.options['random_dst_flip'] = self.load_or_def_option('random_dst_flip', True)

        # Training Params
        self.options['batch_size'] = self.load_or_def_option('batch_size', 16) # Use user default
        self.options['pretrain'] = self.load_or_def_option('pretrain', True)
        self.options['retraining_samples'] = self.load_or_def_option('retraining_samples', False)
        self.options['target_iter'] = self.load_or_def_option('target_iter', 0)

        # Environment/System Params
        self.options['use_fp16'] = self.load_or_def_option('use_fp16', False)

        # Session/Saving Params
        self.options['session_name'] = self.load_or_def_option('session_name', "")
        self.options['autobackup_hour'] = self.load_or_def_option('autobackup_hour', 0)
        self.options['maximum_n_backups'] = self.load_or_def_option('maximum_n_backups', 24)
        self.options['write_preview_history'] = self.load_or_def_option('write_preview_history', False)
        self.options['saving_time'] = self.load_or_def_option('saving_time', 25)

        # Preview Params
        self.options['preview_samples'] = self.load_or_def_option('preview_samples', 2)
        self.options['force_full_preview'] = self.load_or_def_option('force_full_preview', False)

        # Gradient Checkpointing Option
        if force_gradient_checkpointing:
            self.options['gradient_checkpointing'] = True
            io.log_info("Gradient checkpointing forced by command line argument.")
        else:
            # Load saved value, default to True if not saved
            self.options['gradient_checkpointing'] = self.load_or_def_option('gradient_checkpointing', True)

        # --- Ask user interactively if needed ---
        # Use self.silent_start passed from __init__
        ask_override = False if self.read_from_conf else self.ask_override()
        should_ask_interactively = not self.silent_start and (self.is_first_run() or ask_override) and \
                                ((self.read_from_conf and not self.config_file_exists) or not self.read_from_conf)

        if should_ask_interactively:
            io.log_info("Asking for options interactively...")
            # Call ask_* helper methods using the loaded/default values from self.options
            self.ask_session_name(self.options['session_name'])
            self.ask_autobackup_hour(self.options['autobackup_hour'])
            self.ask_maximum_n_backups(self.options['maximum_n_backups'])
            self.ask_write_preview_history(self.options['write_preview_history'])
            self.options['preview_samples'] = np.clip ( io.input_int ("Number of samples to preview", self.options['preview_samples'], add_info="1 - 16"), 1, 16 )
            self.options['force_full_preview'] = io.input_bool ("Use old preview panel", self.options['force_full_preview'])
            self.ask_target_iter(self.options['target_iter'])
            self.ask_retraining_samples(self.options['retraining_samples'])
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(self.options['batch_size'])
            self.options['use_fp16'] = io.input_bool ("Use fp16", self.options['use_fp16'])

            # Ask Architecture Params
            if self.is_first_run() or ask_override:
                self.options['resolution'] = np.clip ( io.input_int("Resolution", self.options['resolution'], add_info="64-640"), 64, 640)
                self.options['face_type'] = io.input_str ("Face type", self.options['face_type'], ['half_face', 'mid_full_face', 'full_face', 'whole_face', 'head']); # Removed custom/mark_only for simplicity
                while True:
                    archi_input = io.input_str ("AE architecture", self.options['archi'], help_message="df/liae -u -d -t -c variants");
                    archi_split = archi_input.lower().split('-');
                    archi_type_input = archi_split[0]; archi_opts_input = archi_split[1] if len(archi_split) >= 2 else '';
                    if archi_type_input in ['df','liae'] and all(opt in 'udtc' for opt in archi_opts_input): self.options['archi'] = archi_input.lower(); break;
                    else: io.log_err("Invalid architecture string.")
                self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", self.options['ae_dims'], add_info="32-1024"), 32, 1024 );
                e_dims_input = np.clip ( io.input_int("Encoder dimensions", self.options['e_dims'], add_info="16-256"), 16, 256 ); self.options['e_dims'] = e_dims_input + e_dims_input % 2;
                d_dims_input = np.clip ( io.input_int("Decoder dimensions", self.options['d_dims'], add_info="16-256"), 16, 256 ); self.options['d_dims'] = d_dims_input + d_dims_input % 2;
                d_mask_dims_input = np.clip ( io.input_int("Decoder mask dimensions", self.options['d_mask_dims'], add_info="16-256"), 16, 256 ); self.options['d_mask_dims'] = d_mask_dims_input + d_mask_dims_input % 2;

            # Ask Training Features
            current_face_type = self.options['face_type']
            if current_face_type in ['whole_face', 'head']: # Simplified check
                self.options['masked_training']  = io.input_bool ("Masked training", self.options['masked_training'])
            else: self.options['masked_training'] = False # Default false for lower face types
            self.options['eyes_mouth_prio'] = io.input_bool ("Eyes and mouth priority", self.options['eyes_mouth_prio'])
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution", self.options['uniform_yaw'])
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", self.options['blur_out_mask'])

            # Ask Optimizer Params
            self.options['adabelief'] = io.input_bool ("Use AdaBelief optimizer?", self.options['adabelief'])
            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", self.options['lr_dropout'], ['n','y','cpu'])
            self.options['lr'] = np.clip (io.input_number("Learning rate", self.options['lr'], add_info="0.0 .. 1.0"), 0.0, 1.0)
            self.options['lr_cos'] = max(0, io.input_int("Cosine LR decay steps", self.options['lr_cos'], add_info="0=off"))
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", self.options['clipgrad'])

            # Ask Loss Params
            self.options['loss_function'] = io.input_str(f"Loss function", self.options['loss_function'], ['SSIM', 'MS-SSIM', 'MS-SSIM+L1'])
            current_archi_type, _ = self._parse_archi()
            if 'df' in current_archi_type: self.options['true_face_power'] = np.clip ( io.input_number ("'True face' power.", self.options['true_face_power'], add_info="0.0000 .. 1.0"), 0.0, 1.0 )
            else: self.options['true_face_power'] = 0.0 # Force 0 for LIAE
            self.options['background_power'] = np.clip ( io.input_number("Background power", self.options['background_power'], add_info="0.0..1.0"), 0.0, 1.0 )
            self.options['face_style_power'] = np.clip ( io.input_number("Face style power", self.options['face_style_power'], add_info="0.0..100.0"), 0.0, 100.0 )
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power", self.options['bg_style_power'], add_info="0.0..100.0"), 0.0, 100.0 )

            # Ask GAN Params
            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", self.options['gan_power'], add_info="0.0 .. 10.0"), 0.0, 10.0 )
            if self.options['gan_power'] > 0.0:
                current_res_for_gan = self.options['resolution']; default_gan_patch_size_interactive = current_res_for_gan // 8
                self.options['gan_patch_size'] = np.clip ( io.input_int("GAN patch size", self.options.get('gan_patch_size', default_gan_patch_size_interactive), add_info="3-640"), 3, 640 )
                self.options['gan_dims'] = np.clip ( io.input_int("GAN dimensions", self.options['gan_dims'], add_info="4-64"), 4, 64 )
                self.options['gan_smoothing'] = np.clip ( io.input_number("GAN label smoothing", self.options['gan_smoothing'], add_info="0 - 0.5"), 0.0, 0.5)
                self.options['gan_noise'] = np.clip ( io.input_number("GAN noisy labels", self.options['gan_noise'], add_info="0 - 0.5"), 0.0, 0.5)

            # Ask Augmentation Params
            self.options['random_warp'] = io.input_bool ("Enable random warp", self.options['random_warp'])
            self.options['random_hsv_power'] = np.clip ( io.input_number ("Random hue/saturation/light intensity", self.options['random_hsv_power'], add_info="0.0 .. 0.3"), 0.0, 0.3 )
            self.options['random_downsample'] = io.input_bool("Enable random downsample", self.options['random_downsample'])
            self.options['random_noise'] = io.input_bool("Enable random noise", self.options['random_noise'])
            self.options['random_blur'] = io.input_bool("Enable random blur", self.options['random_blur'])
            self.options['random_jpeg'] = io.input_bool("Enable random jpeg", self.options['random_jpeg'])
            self.options['random_shadow'] = io.input_str('Enable random shadows', self.options['random_shadow'], ['none','src','dst','all'])
            self.options['ct_mode'] = io.input_str (f"Color transfer mode", self.options['ct_mode'], ['none','rct','lct','mkl','idt','sot', 'fs-aug'])
            self.options['random_color'] = io.input_bool ("Random color", self.options['random_color'])

            # Ask Other Training Params
            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", self.options['pretrain'])
            self.options['gradient_checkpointing'] = io.input_bool ("Enable gradient checkpointing?", self.options['gradient_checkpointing'])

        # --- Final checks and adjustments AFTER interactive session ---
        if self.options.get('pretrain', False):
            pretrain_path = self.get_pretraining_data_path()
            if pretrain_path is None or not pretrain_path.exists():
                io.log_err("Pretraining is enabled, but 'pretraining_data_path' is not valid or not set.")
                self.options['pretrain'] = False; io.log_info("Disabling pretraining mode.")

        archi_type_final, archi_opts_final = self._parse_archi()
        io.log_info(f"DEBUG initialize_options: Parsed archi_type='{archi_type_final}', archi_opts='{archi_opts_final}' for resolution adjustment.") # Added
        min_res, max_res = 64, 640
        current_res = self.options['resolution']
        io.log_info(f"DEBUG initialize_options: User selected/loaded resolution (current_res): {current_res}") # Added
        
        # Adjust resolution based on '-d' opt if present
        res_multiple = 32 if 'd' in archi_opts_final else 16
        io.log_info(f"DEBUG initialize_options: res_multiple based on archi_opts: {res_multiple}") # Added
        
        final_res = np.clip ( ((current_res // res_multiple) * res_multiple), min_res, max_res)
        io.log_info(f"DEBUG initialize_options: Calculated final_res before assignment: {final_res}") # Added
        
        if final_res != current_res: 
            io.log_info(f"Adjusting resolution from {current_res} to {final_res} (multiple of {res_multiple}).")
        self.options['resolution'] = final_res
        io.log_info(f"DEBUG initialize_options: self.options['resolution'] set to: {self.options['resolution']}") # Added

        # Update instance variables from FINAL options
        self.resolution = self.options['resolution']
        self.batch_size = self.options['batch_size']
        self.face_type_str = self.options['face_type']
        try: self.face_type_enum = facelib.FaceType.fromString(self.face_type_str)
        except Exception as e: io.log_err(f"Final check: Error converting face_type '{self.face_type_str}': {e}. Using FULL."); self.face_type_enum = facelib.FaceType.FULL

        io.log_info("SAEHDModel options initialized.")   
        # --- Re-implemented Helper Methods (from old ModelBase or refactoring needs) ---

    def get_strpath_storage_for_file(self, filename):
        """Constructs the full path for saving a model-related file."""
        saved_path = getattr(self, 'saved_models_path', None)
        model_name = self.name
        if saved_path is None or not model_name:
             try: io.log_err("Cannot get storage path: saved_models_path or model name is None/empty.")
             except: print("ERROR: Cannot get storage path: saved_models_path or model name is None/empty.")
             return None
        try:
            model_dir = Path(saved_path) / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir / filename
        except Exception as e:
             try: io.log_err(f"Error creating/accessing model directory {Path(saved_path) / model_name}: {e}")
             except: print(f"ERROR: Error creating/accessing model directory {Path(saved_path) / model_name}: {e}")
             return None

    def load_or_def_option(self, key, default_value):
        """Loads an option from the self.options dict or returns the default."""
        if not hasattr(self, 'options'): self.options = {}
        if key == 'lr_dropout':
             saved_value = self.options.get(key, default_value)
             if isinstance(saved_value, bool): return 'y' if saved_value else 'n'
             return saved_value
        return self.options.get(key, default_value)

    def is_first_run(self):
         """Checks if this is the first run (iteration 0)."""
         iter_val = getattr(self, '_iter_from_save', 0)
         if iter_val == 0 and hasattr(self, 'optimizer_G') and self.optimizer_G is not None:
             # Check optimizer state only if it exists
             if hasattr(self.optimizer_G, 'iterations'):
                 iter_val = self.optimizer_G.iterations.numpy()
             else: # Fallback if iterations attribute isn't ready (e.g., before first apply_gradients)
                 iter_val = 0 # Assume 0 if optimizer state isn't fully ready
         return iter_val == 0

    def ask_override(self):
         """Asks the user if they want to override options, with a timeout."""
         read_from_conf = getattr(self, 'read_from_conf', False)
         silent_start = getattr(self, 'silent_start', False)
         if read_from_conf or self.is_first_run() or silent_start: return False
         try: return io.input_in_time("Press enter in 2 seconds to override model settings.", 2)
         except Exception as e: print(f"Error calling io.input_in_time: {e}. Defaulting to not override."); return False

    def ask_session_name(self, current_value): self.options['session_name'] = io.input_str("Session name", current_value)
    def ask_autobackup_hour(self, current_value): self.options['autobackup_hour'] = np.clip(io.input_int("Autobackup every N hour", current_value, add_info="0..24"), 0, 24)
    def ask_maximum_n_backups(self, current_value): self.options['maximum_n_backups'] = max(0, io.input_int("Maximum N backups", current_value))
    def ask_write_preview_history(self, current_value): self.options['write_preview_history'] = io.input_bool("Write preview history", current_value)
    def ask_target_iter(self, current_value): self.options['target_iter'] = max(0, io.input_int("Target iteration", current_value))
    def ask_retraining_samples(self, current_value): self.options['retraining_samples'] = io.input_bool("Retrain high loss samples", current_value) # May not be used
    def ask_random_src_flip(self): current_value = self.options.get('random_src_flip', False); self.options['random_src_flip'] = io.input_bool("Flip SRC faces randomly", current_value)
    def ask_random_dst_flip(self): current_value = self.options.get('random_dst_flip', True); self.options['random_dst_flip'] = io.input_bool("Flip DST faces randomly", current_value)
    def ask_batch_size(self, current_value): self.options['batch_size'] = max(1, io.input_int("Batch_size", current_value))

    def get_pretraining_data_path(self):
        """Gets the path to pretraining data."""
        return getattr(self, 'pretraining_data_path', None)

    def get_batch_size(self):
        """Gets the current batch size."""
        return getattr(self, 'batch_size', self.options.get('batch_size', 4))

    def is_debug(self):
        """Checks if debug mode is enabled."""
        return getattr(self, 'debug', False)

    def set_training_data_generators(self, generators_list):
         """Sets the initialized sample generators."""
         if isinstance(generators_list, (list, tuple)) and len(generators_list) == 2:
             self.training_data_src, self.training_data_dst = generators_list
             self.generator_list = generators_list
             io.log_info("Training data generators assigned to model.")
         else:
             raise ValueError("set_training_data_generators expects a list/tuple of 2 generators (src, dst).")

    def generate_next_samples(self):
        """Generates the next batch of samples from the assigned generators."""
        if not hasattr(self, 'generator_list') or not self.generator_list or len(self.generator_list) != 2 or self.generator_list[0] is None or self.generator_list[1] is None:
            raise Exception("Sample generators are not initialized or assigned correctly.")
        try:
            # generate_next returns ([sample_arrays], filenames)
            src_data, _ = self.generator_list[0].generate_next()
            dst_data, _ = self.generator_list[1].generate_next()
            # Return unpacked sample arrays
            return (*src_data, *dst_data)
        except Exception as e: io.log_err(f"Error during sample generation: {e}"); traceback.print_exc(); raise e

    def get_iter(self):
        """Gets the current training iteration count from the optimizer."""
        if hasattr(self, 'optimizer_G') and self.optimizer_G and hasattr(self.optimizer_G, 'iterations'):
             return self.optimizer_G.iterations.numpy()
        else: return getattr(self, '_iter_from_save', 0)

    def set_iter(self, iter_val):
        """Sets the training iteration count on optimizers."""
        iter_val = int(iter_val); self._iter_from_save = iter_val
        try:
            # Check if optimizers exist before assigning iterations
            if hasattr(self, 'optimizer_G') and self.optimizer_G: self.optimizer_G.iterations.assign(iter_val)
            if hasattr(self, 'optimizer_D_code') and self.optimizer_D_code: self.optimizer_D_code.iterations.assign(iter_val)
            if hasattr(self, 'optimizer_D_gan') and self.optimizer_D_gan: self.optimizer_D_gan.iterations.assign(iter_val)
        except Exception as e: io.log_err(f"Warning: Error setting optimizer iterations: {e}")

    def get_loss_history(self):
        """Gets the recorded loss history."""
        lh = getattr(self, 'loss_history', [])
        return lh if isinstance(lh, list) else []

    def add_loss_history(self, loss_entry_list):
        """Adds a new entry to the loss history."""
        if not hasattr(self, 'loss_history') or not isinstance(self.loss_history, list): self.loss_history = []
        valid_losses = []
        for l in loss_entry_list:
             if l is not None:
                try: 
                    l_float = float(l);
                    if not np.isnan(l_float) and not np.isinf(l_float): valid_losses.append(l_float)
                except (ValueError, TypeError): pass
        if valid_losses: self.loss_history.append(valid_losses)

    def is_reached_iter_goal(self):
        """Checks if the target iteration has been reached."""
        target_iter = self.get_target_iter(); return target_iter != 0 and self.get_iter() >= target_iter

    def get_target_iter(self):
        """Gets the target iteration from options."""
        return self.options.get('target_iter', 0)

    def save_model_data(self):
        """Saves options, loss history, and preview samples."""
        if self.model_data_path is None: io.log_err("Model data path is None."); return;
        try:
            model_data = {
                'iter': self.get_iter(),
                'options': self.options,
                'loss_history': self.get_loss_history(),
                'sample_for_preview': getattr(self, 'sample_for_preview', None)
             }
            self.model_data_path.write_bytes(pickle.dumps(model_data))
        except Exception as e: io.log_err(f"Error saving model data: {e}")

    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        """Updates the samples used for generating previews."""
        if not self.is_training: return
        preview_samples_count = self.options.get('preview_samples', 2) # Use updated default
        new_samples = []; new_sample_filenames = []
        if hasattr(self, 'generator_list') and self.generator_list and self.generator_list[0] is not None and self.generator_list[1] is not None:
            try:
                for i in range(preview_samples_count):
                     # Generate a batch and take the i-th sample's targets
                     samples_data = self.generate_next_samples() # Gets 8 tensors
                     # Need to handle potential batch size differences if bs < preview_samples_count
                     batch_idx = i % self.get_batch_size() # Index within the generated batch
                     target_s = samples_data[1][batch_idx]; target_d = samples_data[5][batch_idx] # Target indices
                     new_samples.append((target_s, target_d))
                     # Filename handling requires generator changes - skip for now
                     new_sample_filenames.append((None, None))
            except StopIteration: io.log_info("Sample generator exhausted during preview update.");
            except Exception as e: io.log_err(f"Error generating sample {i} for preview: {e}"); traceback.print_exc(); return # Stop update on error
            if new_samples: self.sample_for_preview = new_samples; self.sample_filenames = new_sample_filenames; io.log_info("Preview samples updated.")
        else: io.log_info("Warning: Cannot update preview samples: Sample generators not available.")

    def get_preview_samples(self):
        """Returns the stored samples designated for preview generation."""
        if not hasattr(self, 'sample_for_preview') or self.sample_for_preview is None or len(self.sample_for_preview) == 0:
             io.log_info("Preview samples are None/empty, attempting to update...")
             self.update_sample_for_preview(force_new=True)
             if not hasattr(self, 'sample_for_preview') or self.sample_for_preview is None or len(self.sample_for_preview) == 0:
                  io.log_err("Failed to update preview samples. Cannot provide preview data.")
                  return None
        return self.sample_for_preview

    def onGetPreview(self, samples_data):
        """Generates preview images from the provided sample data."""
        io.log_info(f"DEBUG onGetPreview: self.resolution for preview image assembly: {self.resolution}") # Added
        if samples_data is None or len(samples_data) == 0: return []
        preview_images = []
        num_samples_to_show = min(len(samples_data), self.options.get('preview_samples', 2))
        if num_samples_to_show == 0: return []
        batch_src_t = []; batch_dst_t = []
        filenames_to_use = getattr(self, 'sample_filenames', [(None, None)] * num_samples_to_show)
        for i in range(num_samples_to_show):
            src_sample_np, dst_sample_np = samples_data[i]
            batch_src_t.append(src_sample_np); batch_dst_t.append(dst_sample_np)
        try: batch_src_t = tf.convert_to_tensor(batch_src_t, dtype=tf.keras.backend.floatx()); batch_dst_t = tf.convert_to_tensor(batch_dst_t, dtype=tf.keras.backend.floatx())
        except Exception as e: io.log_err(f"Error converting preview samples to tensors: {e}"); return []
        try:
            outputs = self.call([batch_src_t, batch_dst_t], training=True) # Use training=True to get all paths
            archi_type_final, _ = self._parse_archi()
            # Assuming LIAE architecture based on options used:
            p_s_s_b, p_s_sm_b, p_d_d_b, p_d_dm_b, p_s_d_b, p_s_dm_b, _, _ = outputs

            S_b = batch_src_t.numpy(); D_b = batch_dst_t.numpy();
            SS_b = p_s_s_b.numpy(); DD_b = p_d_d_b.numpy(); SD_b = p_s_d_b.numpy() # src->dst prediction

            for i in range(num_samples_to_show):
                 S, D, SS, DD, SD = S_b[i], D_b[i], SS_b[i], DD_b[i], SD_b[i]
                 # Normalize predictions - check range first
                 # print(f"DEBUG Sample {i}: SD min/max before clip: {np.min(SD):.4f}/{np.max(SD):.4f}")
                 S, D, SS, DD, SD = [(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) for img in [S, D, SS, DD, SD]]

                 try:
                     # Layout: [[S, SD], [D, DD]]
                     if not (S.shape == D.shape == SD.shape == DD.shape): raise ValueError(f"Mismatched shapes")
                     top_row = cv2.hconcat([S, SD]); bottom_row = cv2.hconcat([D, DD]); combined_image = cv2.vconcat([top_row, bottom_row])
                 except Exception as e_concat: io.log_err(f"Error during preview hconcat/vconcat: {e_concat}"); combined_image = S if S is not None else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

                 title = f"Preview {i}";
                 if len(filenames_to_use) > i: sf, df = filenames_to_use[i];
                 if sf and df: title += f" (S: {Path(sf).name}, D: {Path(df).name})"
                 elif sf: title += f" (S: {Path(sf).name})"
                 elif df: title += f" (D: {Path(df).name})"
                 preview_images.append( (title, combined_image) )
        except Exception as e: io.log_err(f"Error during inference or image assembly in onGetPreview: {e}"); traceback.print_exc(); return []
        return preview_images

    def initialize_sample_generators(self):
        """Initializes sample generators for training."""
        if self.is_training and not self.is_exporting:
            # --- Add print for resolution USED BY GENERATOR ---
            sg_resolution = self.options['resolution']
            io.log_info(f"DEBUG initialize_sample_generators: Using resolution for generators: {sg_resolution}")
            # -------------------------------------------------
            io.log_info(f"DEBUG INIT_GEN: self.face_type_enum = {repr(self.face_type_enum)}")
            training_data_src_path = self.training_data_src_path; training_data_dst_path = self.training_data_dst_path; pretraining_data_path = self.get_pretraining_data_path();
            io.log_info(f"DEBUG INIT_GEN: SRC Path: {training_data_src_path} | Exists: {training_data_src_path.exists()}")
            io.log_info(f"DEBUG INIT_GEN: DST Path: {training_data_dst_path} | Exists: {training_data_dst_path.exists()}")
            random_ct_samples_path = None # Disable CT path for now
            if not training_data_src_path.exists(): raise ValueError(f'Training data src directory does not exist: {training_data_src_path}')
            if not training_data_dst_path.exists(): raise ValueError(f'Training data dst directory does not exist: {training_data_dst_path}')
            if self.options.get('pretrain', False) and (pretraining_data_path is None or not pretraining_data_path.exists()): raise ValueError(f'Pretraining data directory does not exist: {pretraining_data_path}')
            resolution = sg_resolution; face_type = self.options['face_type'];
            io.log_info(f"DEBUG: face_type = {face_type}, self.face_type_str = {self.face_type_str}")
            try:
                 augs = samplelib.SampleProcessor.Options.AugmentationParams( random_flip=False, ct_mode=self.options.get('ct_mode', 'none'), random_ct_mode=self.options.get('random_color', False) and not self.options.get('pretrain', False), random_hsv_power=self.options.get('random_hsv_power', 0.0) if not self.options.get('pretrain', False) else 0.0, random_downsample=self.options.get('random_downsample', False) and not self.options.get('pretrain', False), random_noise=self.options.get('random_noise', False) and not self.options.get('pretrain', False), random_blur=self.options.get('random_blur', False) and not self.options.get('pretrain', False), random_jpeg=self.options.get('random_jpeg', False) and not self.options.get('pretrain', False), random_shadow=self.options.get('random_shadow', 'none') if not self.options.get('pretrain', False) else 'none')
            except AttributeError: io.log_err("FATAL: Could not access SampleProcessor.Options.AugmentationParams."); raise
            try: sample_process_options = samplelib.SampleProcessor.Options( batch_size=self.get_batch_size(), resolution=self.resolution, face_type=self.face_type_enum, mask_type=samplelib.SampleProcessor.FaceMaskType.FULL_FACE if self.options.get('masked_training', True) else samplelib.SampleProcessor.FaceMaskType.NONE, eye_mouth_prio=self.options.get('eyes_prio', False) or self.options.get('mouth_prio', False), augmentation_params=augs, random_warp=False, true_face_power=self.options.get('true_face_power',0.0), random_flip=True )
            except AttributeError as e: io.log_err(f"FATAL: Error accessing SampleProcessor nested class/enum - {e}."); raise
            except Exception as e: io.log_err(f"Error creating SampleProcessor.Options: {e}"); traceback.print_exc(); self.generator_list = None; return
            SampleProcessor_sample_options = sample_process_options
            SampleProcessor_sample_dst_options = copy.deepcopy(sample_process_options)
            SampleProcessor_sample_dst_options.random_flip = self.options.get('random_dst_flip', True)
            SampleProcessor_sample_dst_options.random_warp = False # <--- TEMPORARILY DISABLED also for DST options
            SampleProcessor_sample_options.random_flip = self.options.get('random_src_flip', False)
            SampleProcessor_sample_options.random_warp = False # <--- TEMPORARILY DISABLED also for SRC options
            output_sample_types_defs = [ {'sample_type': samplelib.SampleProcessor.SampleType.FACE_IMAGE, 'warp': True, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.BGR, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, {'sample_type': samplelib.SampleProcessor.SampleType.FACE_IMAGE, 'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.BGR, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, {'sample_type': samplelib.SampleProcessor.SampleType.FACE_MASK, 'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.G, 'face_mask_type': samplelib.SampleProcessor.FaceMaskType.FULL_FACE, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, {'sample_type': samplelib.SampleProcessor.SampleType.FACE_MASK, 'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.G, 'face_mask_type': samplelib.SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution} ]
            generators_count = min(4, multiprocessing.cpu_count());
            if self.options.get('pretrain', False): io.log_info(f"Initializing SampleGeneratorFace for PRETRAINING using {pretraining_data_path}"); src_dst_generators = [ samplelib.SampleGeneratorFace( pretraining_data_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False) or self.options.get('pretrain', False), retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count) ] * generators_count; self.set_training_data_generators([src_dst_generators[0], src_dst_generators[0]])
            else: io.log_info(f"Initializing SampleGeneratorFace for SRC using {training_data_src_path}"); src_generators = [ samplelib.SampleGeneratorFace( training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False), retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count ) ] * generators_count; io.log_info(f"Initializing SampleGeneratorFace for DST using {training_data_dst_path}"); dst_generators = [ samplelib.SampleGeneratorFace( training_data_dst_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_dst_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False), retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count ) ] * generators_count; self.set_training_data_generators([src_generators[0], dst_generators[0]])
        else: self.generator_list = None

    # --- End of Initializers ---

    def predictor_func(self, face=None, **kwargs):
        """Performs inference for merging."""
        if face is None: raise ValueError("Predictor received None face.");
        # Ensure input is float32, add batch dim
        face_t = tf.convert_to_tensor(face[None,...], dtype=tf.float32);
        # Call model with training=False for inference
        p_f_t, p_s_dm_t, p_d_dm_t = self.call(face_t, training=False);
        # Return face and both masks (src->dst mask, dst->dst mask)
        return np.clip(p_f_t[0].numpy(), 0, 1), np.clip(p_s_dm_t[0].numpy(), 0, 1), np.clip(p_d_dm_t[0].numpy(), 0, 1);

    def get_MergerConfig(self):
        """Returns configuration needed for the merger script."""
        import merger # Local import for safety
        # Determine if morphing is applicable based on archi_type
        archi_type = getattr(self, 'archi_type', self._parse_archi()[0]) # Get archi_type safely
        is_morphable = ('liae' in archi_type) # Only LIAE supports morphing in original
        # Return predictor function, resolution tuple, and MergerConfigMasked
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=self.face_type_enum, is_morphable=is_morphable)

# --- START OF FILE models/Model_SAEHD/Model.py --- (Part 2/3)

# (Continued from Part 1)
# Inside class Model(tf.keras.Model):

    def _parse_archi(self):
        """Helper to parse architecture string from self.options."""
        archi_str = self.options.get('archi', '') # Safely get archi string
        archi_split = archi_str.split('-')
        archi_type = archi_split[0].lower() if len(archi_split) > 0 else ''
        # Handle cases with or without opts correctly
        archi_opts = archi_split[1].lower() if len(archi_split) >= 2 else ''
        return archi_type, archi_opts

    # --- call method ---
    def call(self, inputs, training=False):
        # --- Ensure Checkpointing is Disabled ---
        use_cp = False # Force disabled
        # ---------------------------------------

        # --- Define helper functions (no recompute_grad needed) ---
        def r_enc(layer_input, training): return self.encoder(layer_input, training=training)
        def r_int(layer_input, training): return self.inter(layer_input, training=training)
        def r_intA(layer_input, training): return self.inter_AB(layer_input, training=training)
        def r_intB(layer_input, training): return self.inter_B(layer_input, training=training)
        def r_dec(layer_input, training): return self.decoder(layer_input, training=training)
        def r_decS(layer_input, training): return self.decoder_src(layer_input, training=training)
        def r_decD(layer_input, training): return self.decoder_dst(layer_input, training=training)
        
        # Use helpers directly
        enc_op = r_enc
        int_op = r_int
        intA_op = r_intA
        intB_op = r_intB
        dec_op = r_dec
        decS_op = r_decS
        decD_op = r_decD

        # --- Input Handling ---
        is_inference = False
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            warped_src, warped_dst = inputs
            if warped_src is None or warped_dst is None: raise ValueError("Training input requires both warped_src and warped_dst.")
        elif isinstance(inputs, tf.Tensor):
            warped_dst = inputs; warped_src = None; is_inference = True
            if training: io.log_info("Warning: Model called with training=True but received single tensor input."); training = False
        else: raise ValueError(f"Unexpected input type to call: {type(inputs)}.")

        # Override inference flag if model is explicitly in training mode
        if self.is_training: is_inference = False; training = True

        # --- Encoding ---
        if not hasattr(self, 'encoder'): raise AttributeError("Model encoder not initialized.")
        src_code_enc = enc_op(warped_src, training) if warped_src is not None else None
        dst_code_enc = enc_op(warped_dst, training)
        # ---- DEBUG Encoder Output ----
        if dst_code_enc is not None: # Check since it could be None in some paths if not careful
            tf.print("DEBUG Model.call: dst_code_enc (Encoder output for Inter) min/max/mean/std:", 
                     tf.reduce_min(dst_code_enc), tf.reduce_max(dst_code_enc), 
                     tf.reduce_mean(dst_code_enc), tf.math.reduce_std(dst_code_enc), 
                     output_stream=sys.stdout, summarize=-1) # summarize to see more if it's large
        if src_code_enc is not None:
            tf.print("DEBUG Model.call: src_code_enc (Encoder output for Inter) min/max/mean/std:", 
                     tf.reduce_min(src_code_enc), tf.reduce_max(src_code_enc), 
                     tf.reduce_mean(src_code_enc), tf.math.reduce_std(src_code_enc), 
                     output_stream=sys.stdout, summarize=-1)
        # ----------------------------

        # --- Intermediate and Decoding (based on architecture type) ---
        archi_type_final, _ = self._parse_archi()

        if 'df' in archi_type_final:
            if warped_src is None and not is_inference: raise ValueError("DF training requires src input.")
            if not hasattr(self, 'inter') or not hasattr(self, 'decoder_src') or not hasattr(self, 'decoder_dst'): raise AttributeError("DF components not initialized.")

            g_src_c = int_op(src_code_enc, training) if src_code_enc is not None else None
            g_dst_c = int_op(dst_code_enc, training)

            # Dummy shapes for inference if src is None (needed for consistent return signature)
            # Note: compute_dtype might not be reliable if mixed precision isn't fully set up
            # Use backend floatx or explicitly tf.float32/tf.float16
            tensor_dtype = tf.keras.backend.floatx()
            dummy_shape_bgr = tf.shape(g_dst_c); dummy_shape_mask = tf.concat([tf.shape(g_dst_c)[:-1], [1]], axis=0)

            p_s_s, p_s_sm = decS_op(g_src_c, training) if g_src_c is not None else (tf.zeros(dummy_shape_bgr, dtype=tensor_dtype), tf.zeros(dummy_shape_mask, dtype=tensor_dtype))
            p_d_d, p_d_dm = decD_op(g_dst_c, training)
            p_s_d, p_s_dm = decS_op(g_dst_c, training)

            if not is_inference: return p_s_s, p_s_sm, p_d_d, p_d_dm, p_s_d, p_s_dm, g_src_c, g_dst_c
            else: return p_s_d, p_s_dm, p_d_dm

        elif 'liae' in archi_type_final:
            if not hasattr(self, 'inter_AB') or not hasattr(self, 'inter_B') or not hasattr(self, 'decoder'): raise AttributeError("LIAE components not initialized.")

            g_dst_iAB = intA_op(dst_code_enc, training)
            g_dst_iB = intB_op(dst_code_enc, training)

            dfmt = tf.keras.backend.image_data_format(); c_ax = -1 if dfmt == 'channels_last' else 1
            g_dst_c = tf.concat([g_dst_iB, g_dst_iAB], axis=c_ax)
            
            # For src->dst, we need source identity but g_src_iAB may not be defined yet (especially during inference)
            # So first make sure we calculate it if we're in training mode
            if not is_inference and src_code_enc is not None:
                g_src_iAB = intA_op(src_code_enc, training)
                g_sd_c = tf.concat([g_dst_iB, g_src_iAB], axis=c_ax) # Correct: dst expression with src identity
            else:
                # During inference, fall back to using destination identity
                g_sd_c = tf.concat([g_dst_iB, g_dst_iAB], axis=c_ax) # Fallback for inference

            p_d_d, p_d_dm = dec_op(g_dst_c, training)
            p_s_d, p_s_dm = dec_op(g_sd_c, training)

            if not is_inference:
                if warped_src is None: raise ValueError("LIAE training requires src input.")
                if src_code_enc is None: raise ValueError("LIAE training encoder output for src is None.")
                # g_src_iAB was already calculated above
                # Now create Src -> Src input by combining src identity with itself
                g_src_c = tf.concat([g_src_iAB, g_src_iAB], axis=c_ax)
                p_s_s, p_s_sm = dec_op(g_src_c, training)
                return p_s_s, p_s_sm, p_d_d, p_d_dm, p_s_d, p_s_dm, g_src_c, g_dst_c
            else: 
                # Return just what's needed for inference
                return p_s_d, p_s_dm, p_d_dm
        else:
            raise ValueError(f"Unknown architecture type in call: {archi_type_final}")

    # train_step (No @tf.function here)
    def train_step(self, data): # Removed apply_G_gradients parameter
        """
        Performs one training step using tf.GradientTape.
        """
        # --- Unpack Data ---
        try:
            warped_src, target_src, target_srcm_all, target_srcm_em, \
            warped_dst, target_dst, target_dstm_all, target_dstm_em = data
        except ValueError:
             io.log_err("Error unpacking data in train_step. Check SampleGenerator output.");
             loss_zero = tf.constant(0.0, dtype=tf.float32)
             # Return structure expected by Trainer.py loss aggregation
             return {'loss_G_total': loss_zero, 'loss_G_src': loss_zero, 'loss_G_dst': loss_zero,
                     'loss_G_gan': loss_zero, 'loss_G_true_face': loss_zero,
                     'loss_D_gan': loss_zero, 'loss_D_code': loss_zero}

        # --- Generator Training Step ---
        grads_G = None; grads_D_gan = None; grads_D_code = None; # Initialize grads
        loss_G_total = tf.constant(0.0, dtype=tf.float32)
        loss_G_src = tf.constant(0.0, dtype=tf.float32)
        loss_G_dst = tf.constant(0.0, dtype=tf.float32)
        loss_G_gan = tf.constant(0.0, dtype=tf.float32)
        loss_G_true_face = tf.constant(0.0, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as g_tape:
            # === FORWARD PASS FIRST ===
            outputs = self.call([warped_src, warped_dst], training=True)

            # === THEN COLLECT VARIABLES ===
            G_vars = []; D_code_vars = []; D_gan_vars = []
            archi_type_final, _ = self._parse_archi()
            if 'df' in archi_type_final:
                if hasattr(self,'encoder'): G_vars.extend(self.encoder.trainable_variables)
                if hasattr(self,'inter'): G_vars.extend(self.inter.trainable_variables)
                if hasattr(self,'decoder_src'): G_vars.extend(self.decoder_src.trainable_variables)
                if hasattr(self,'decoder_dst'): G_vars.extend(self.decoder_dst.trainable_variables)
            elif 'liae' in archi_type_final:
                if hasattr(self,'encoder'): G_vars.extend(self.encoder.trainable_variables)
                if hasattr(self,'inter_AB'): G_vars.extend(self.inter_AB.trainable_variables)
                if hasattr(self,'inter_B'): G_vars.extend(self.inter_B.trainable_variables)
                if hasattr(self,'decoder'): G_vars.extend(self.decoder.trainable_variables)

            # Discriminator vars (currently discriminators are disabled)
            # if self.code_discriminator: D_code_vars = self.code_discriminator.trainable_variables
            # if self.D_src: D_gan_vars = self.D_src.trainable_variables
            # ============================

            # --- Unpack outputs AFTER collecting variables ---
            # Make sure unpacking matches the return signature of call() for the specific archi
            if 'df' in archi_type_final or 'liae' in archi_type_final:
                p_s_s, p_s_sm, p_d_d, p_d_dm, p_s_d, p_s_dm, g_src_c, g_dst_c = outputs
            else: raise ValueError("Unsupported architecture in train_step G pass.")

            # --- Calculate Generator Losses ---
            loss_G_src = self._calculate_recon_loss(target_src, target_srcm_all, p_s_s, p_s_sm, target_srcm_em)
            loss_G_dst = self._calculate_recon_loss(target_dst, target_dstm_all, p_d_d, p_d_dm, target_dstm_em)
            loss_G_total += loss_G_src + loss_G_dst

            # ---- DEBUG NaNs in Loss ----
            tf.print("DEBUG train_step: loss_G_src:", loss_G_src, "loss_G_dst:", loss_G_dst, "loss_G_total:", loss_G_total, output_stream=sys.stdout)
            loss_G_total = tf.debugging.check_numerics(loss_G_total, "loss_G_total calculation")
            # -----------------------------

            # --- GAN Loss (Currently Disabled) ---
            gan_power = self.options.get('gan_power', 0.0)
            # if gan_power > 0.0 and self.D_src is not None: ...

            # --- True Face Loss (Currently Disabled) ---
            true_face_power = self.options.get('true_face_power', 0.0)
            # if 'df' in archi_type_final and true_face_power > 0.0 and self.code_discriminator is not None: ...

            # --- Style Loss (Placeholder) ---
            face_style_power = self.options.get('face_style_power', 0.0)
            bg_style_power = self.options.get('bg_style_power', 0.0)
            # if face_style_power > 0.0 or bg_style_power > 0.0:
            #     loss_style = self._calculate_style_loss(...) # Needs implementation
            #     loss_G_total += loss_style

        # === Compute Generator Gradients ===
        if G_vars: # Check if G_vars list is populated
             grads_G = g_tape.gradient(loss_G_total, G_vars)
             # ---- DEBUG NaNs in Gradients ----
             if grads_G is not None and G_vars is not None and len(grads_G) == len(G_vars): # Ensure lists are valid
                 checked_grads_G = []
                 for i, g in enumerate(grads_G):
                     if g is not None:
                         # Use sys.stdout for tf.print to ensure it appears in typical console redirection
                         tf.print(f"DEBUG train_step: Grad for {G_vars[i].name}", tf.reduce_sum(g), tf.math.is_nan(tf.reduce_sum(g)), output_stream=sys.stdout)
                         checked_grads_G.append(tf.debugging.check_numerics(g, f"Gradient for G_vars[{i}] ({G_vars[i].name})"))
                     else:
                         tf.print(f"DEBUG train_step: Gradient for var {G_vars[i].name} is None", output_stream=sys.stdout)
                         checked_grads_G.append(None) # Keep None grads as None
                 grads_G = checked_grads_G # Assign back the (potentially) checked gradients
             elif grads_G is None:
                 tf.print("DEBUG train_step: grads_G is None after tape.gradient() call.", output_stream=sys.stdout)
             # ---------------------------------
        else:
             io.log_err("Warning: No Generator variables found to compute gradients.")
             grads_G = None

        # --- Discriminator Training Steps (Currently Disabled) ---
        loss_D_gan_total = tf.constant(0.0, dtype=tf.float32)
        loss_D_code_total = tf.constant(0.0, dtype=tf.float32)
        # if gan_power > 0.0 and self.D_src is not None and self.optimizer_D_gan: ...
        # if 'df' in archi_type_final and true_face_power > 0.0 and self.code_discriminator is not None and self.optimizer_D_code: ...

        # --- Apply Gradients ---
        # Apply G gradients (LR dropout condition removed for now)
        if self.optimizer_G and grads_G is not None:
             # Check if gradients are all None (can happen if loss doesn't depend on vars)
             if not all(g is None for g in grads_G):
                 if self.options.get('clipgrad', False): grads_G, _ = tf.clip_by_global_norm(grads_G, 1.0)
                 self.optimizer_G.apply_gradients(zip(grads_G, G_vars))
             else:
                  io.log_info("Warning: Generator gradients are None, skipping apply_gradients.")

        # Apply D gradients (currently disabled)
        # if self.optimizer_D_gan and grads_D_gan is not None: ...
        # if self.optimizer_D_code and grads_D_code is not None: ...

        # --- Clean up persistent tape ---
        del g_tape

        # --- Return Scalar Losses for Logging ---
        return {
            'loss_G_total': loss_G_total,
            'loss_G_src': loss_G_src,
            'loss_G_dst': loss_G_dst,
            'loss_G_gan': loss_G_gan, # Will be 0
            'loss_G_true_face': loss_G_true_face, # Will be 0
            'loss_D_gan': loss_D_gan_total, # Will be 0
            'loss_D_code': loss_D_code_total # Will be 0
        }

    # --- Helper Loss Functions ---
    @staticmethod
    def _gan_generator_loss(fake_output_logits):
        # Standard GAN generator loss (wants discriminator to predict 1)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_output_logits), fake_output_logits))

    @staticmethod
    def _gan_discriminator_loss(real_output_logits, fake_output_logits):
        # Standard GAN discriminator loss (wants 1 for real, 0 for fake)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_output_logits), real_output_logits))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_output_logits), fake_output_logits))
        return (real_loss + fake_loss) * 0.5

    def _calculate_recon_loss(self, target_img, target_mask_all, pred_img, pred_mask, target_mask_em_combined):
        """
        Calculates the reconstruction loss based on DFL SAEHD logic.
        Includes EM mask unpacking and resolution-dependent SSIM.
        """
        total_loss = tf.constant(0.0, dtype=tf.float32)

        # --- Get Options ---
        resolution = self.options['resolution']
        masked_training = self.options.get('masked_training', True)
        # Combine original eye/mouth flags for simplicity in check
        eyes_mouth_prio_enabled = self.options.get('eyes_prio', False) or self.options.get('mouth_prio', False)
        loss_func_str = self.options.get('loss_function', 'SSIM')
        background_power = self.options.get('background_power', 0.0) # Get background power

        # --- Unpack Combined Eye/Mouth Mask if needed ---
        target_mask_prio = None
        if eyes_mouth_prio_enabled:
            # Based on original logic: 0=bg, 1=face, 2=eyes, 3=mouth
            # Ensure target_mask_em_combined is float for calculations
            target_mask_em_combined_f = tf.cast(target_mask_em_combined, tf.float32)
            target_mask_eye_mouth_area = tf.clip_by_value(target_mask_em_combined_f - 1.0, 0.0, 1.0)
            target_mask_mouth_area = tf.clip_by_value(target_mask_em_combined_f - 2.0, 0.0, 1.0)
            target_mask_eyes_area = tf.clip_by_value(target_mask_eye_mouth_area - target_mask_mouth_area, 0.0, 1.0)

            # Select the priority mask based on original separate options if they still exist, or the combined one
            eyes_prio_opt = self.options.get('eyes_prio', False)
            mouth_prio_opt = self.options.get('mouth_prio', False)
            if eyes_prio_opt and mouth_prio_opt: target_mask_prio = target_mask_eye_mouth_area
            elif eyes_prio_opt: target_mask_prio = target_mask_eyes_area
            elif mouth_prio_opt: target_mask_prio = target_mask_mouth_area
            # If only the combined flag 'eyes_mouth_prio' exists, use target_mask_eye_mouth_area
            elif self.options.get('eyes_mouth_prio', False): target_mask_prio = target_mask_eye_mouth_area


        # --- Mask Blurring (Skipped for now - apply if needed later) ---
        target_mask_all_blur = target_mask_all
        target_mask_all_anti_blur = 1.0 - target_mask_all_blur # Used for bg loss

        # --- Apply Masking for Main Loss (Optional) ---
        if masked_training:
            target_img_masked = target_img * target_mask_all_blur
            pred_img_masked = pred_img * target_mask_all_blur
        else:
            target_img_masked = target_img
            pred_img_masked = pred_img

        # --- DSSIM/MS-SSIM Loss ---
        loss_term_1 = tf.constant(0.0, dtype=tf.float32) # SSIM/MSSIM part
        loss_term_2 = tf.constant(0.0, dtype=tf.float32) # L1 part

        if loss_func_str == 'MS-SSIM' or loss_func_str == 'MS-SSIM+L1':
            if MSSimLoss is not None:
                # Need to instantiate MS-SSIM correctly, check required args
                # Assuming it takes max_val at least
                ms_ssim_loss_obj = MSSimLoss(max_val=1.0) # Add other args if needed
                per_sample_loss = ms_ssim_loss_obj(target_img_masked, pred_img_masked)
                loss_term_1 = 10.0 * tf.reduce_mean(per_sample_loss)
                if loss_func_str == 'MS-SSIM+L1':
                    loss_term_2 = 10.0 * tf.reduce_mean(tf.square(target_img_masked - pred_img_masked)) # Use square like original
            else:
                io.log_err("MSSim Loss class not available! Falling back to L1.")
                loss_term_2 = 10.0 * tf.reduce_mean(tf.square(target_img_masked - pred_img_masked))
        else: # Default to SSIM + L1 Squared
            if DssimLoss is not None:
                # Use filter_size based on resolution, like original
                fs = int(resolution / 11.6)
                dssim_loss_obj = DssimLoss(max_val=1.0, filter_size=fs)
                per_sample_loss_dssim = dssim_loss_obj(target_img_masked, pred_img_masked)
                loss_term_1 = 10.0 * tf.reduce_mean(per_sample_loss_dssim)

                if resolution >= 256: # Add second DSSIM term for high resolution
                    fs_hr = int(resolution / 23.2)
                    dssim_loss_obj_hr = DssimLoss(max_val=1.0, filter_size=fs_hr)
                    per_sample_loss_dssim_hr = dssim_loss_obj_hr(target_img_masked, pred_img_masked)
                    # Original weighted each by 5, totaling 10
                    loss_term_1 = 5.0 * tf.reduce_mean(per_sample_loss_dssim) + 5.0 * tf.reduce_mean(per_sample_loss_dssim_hr)
            else:
                io.log_err("DSSIM Loss class not available! Using only L1.")
                loss_term_1 = tf.constant(0.0, dtype=tf.float32)

                # Always add Squared L1 for SSIM mode
                loss_term_2 = 10.0 * tf.reduce_mean(tf.square(target_img_masked - pred_img_masked))

        total_loss += loss_term_1 + loss_term_2

        # --- Eye/Mouth Priority Loss ---
        if eyes_mouth_prio_enabled and target_mask_prio is not None:
            # Ensure masks have compatible shapes/types before multiplication
            target_mask_prio_f = tf.cast(target_mask_prio, target_img.dtype)
            target_em_masked = target_img * target_mask_prio_f
            pred_em_masked = pred_img * target_mask_prio_f
            loss_em_prio = 300.0 * tf.reduce_mean(tf.abs(target_em_masked - pred_em_masked))
            total_loss += loss_em_prio

        # --- Mask Reconstruction Loss ---
        # Ensure masks have compatible types
        target_mask_all_f = tf.cast(target_mask_all, pred_mask.dtype)
        loss_mask_recon = 10.0 * tf.reduce_mean(tf.square(target_mask_all_f - pred_mask))
        total_loss += loss_mask_recon

        # --- Background Reconstruction Loss ---
        # Original code added this regardless of masked_training, let's replicate
        if background_power > 0.0:
             # Loss on the full image
            if DssimLoss is not None:
                fs_bg = int(resolution / 11.6)
                dssim_loss_obj_bg = DssimLoss(max_val=1.0, filter_size=fs_bg)
                per_sample_loss_dssim_bg = dssim_loss_obj_bg(target_img, pred_img)
                loss_bg_dssim = 10.0 * tf.reduce_mean(per_sample_loss_dssim_bg)
                if resolution >= 256: # Split for high res
                    fs_hr_bg = int(resolution / 23.2)
                    dssim_loss_obj_hr_bg = DssimLoss(max_val=1.0, filter_size=fs_hr_bg)
                    per_sample_loss_dssim_hr_bg = dssim_loss_obj_hr_bg(target_img, pred_img)
                    loss_bg_dssim = 5.0 * tf.reduce_mean(per_sample_loss_dssim_bg) + 5.0 * tf.reduce_mean(per_sample_loss_dssim_hr_bg)
                total_loss += loss_bg_dssim * background_power
            else: # Fallback L1 squared for background
                total_loss += 10.0 * tf.reduce_mean(tf.square(target_img - pred_img)) * background_power

            # Original also added squared L1 for background
            loss_bg_l1_sq = 10.0 * tf.reduce_mean(tf.square(target_img - pred_img))
            total_loss += loss_bg_l1_sq * background_power

        # Ensure loss is finite
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.0, dtype=tf.float32))
        return total_loss

    def _calculate_style_loss(self, pred_src_dst, pred_dst_dst, target_dst, target_dstm_style_blur, target_dstm_style_anti_blur):
        """Placeholder for Style Loss."""
        # Requires VGG or similar feature extractor
        # io.log_info("Warning: _calculate_style_loss needs full DFL implementation!")
        return tf.constant(0.0, dtype=tf.float32)

    def initialize_sample_generators(self):
        """Initializes sample generators for training."""
        if self.is_training and not self.is_exporting:
            io.log_info(f"DEBUG INIT_GEN: self.face_type_enum = {repr(self.face_type_enum)}")
            training_data_src_path = self.training_data_src_path; training_data_dst_path = self.training_data_dst_path; pretraining_data_path = self.get_pretraining_data_path();
            io.log_info(f"DEBUG INIT_GEN: SRC Path: {training_data_src_path} | Exists: {training_data_src_path.exists()}")
            io.log_info(f"DEBUG INIT_GEN: DST Path: {training_data_dst_path} | Exists: {training_data_dst_path.exists()}")
            random_ct_samples_path = None # Disable CT path for now
            if not training_data_src_path.exists(): raise ValueError(f'Training data src directory does not exist: {training_data_src_path}')
            if not training_data_dst_path.exists(): raise ValueError(f'Training data dst directory does not exist: {training_data_dst_path}')
            # Check pretrain path only if pretrain option is actually enabled
            pretrain_enabled = self.options.get('pretrain', False)
            if pretrain_enabled and (pretraining_data_path is None or not pretraining_data_path.exists()):
                raise ValueError(f'Pretraining is enabled but pretraining data directory does not exist: {pretraining_data_path}')

            resolution = self.options['resolution']; face_type = self.options['face_type'];
            io.log_info(f"DEBUG: face_type = {face_type}, self.face_type_str = {self.face_type_str}")
            try:
                 augs = samplelib.SampleProcessor.Options.AugmentationParams( random_flip=False, ct_mode=self.options.get('ct_mode', 'none'), random_ct_mode=self.options.get('random_color', False) and not pretrain_enabled, random_hsv_power=self.options.get('random_hsv_power', 0.0) if not pretrain_enabled else 0.0, random_downsample=self.options.get('random_downsample', False) and not pretrain_enabled, random_noise=self.options.get('random_noise', False) and not pretrain_enabled, random_blur=self.options.get('random_blur', False) and not pretrain_enabled, random_jpeg=self.options.get('random_jpeg', False) and not pretrain_enabled, random_shadow=self.options.get('random_shadow', 'none') if not pretrain_enabled else 'none')
            except AttributeError: io.log_err("FATAL: Could not access SampleProcessor.Options.AugmentationParams."); raise
            try:
                # Determine mask type based on masked_training option
                mask_type_enum = samplelib.SampleProcessor.FaceMaskType.FULL_FACE if self.options.get('masked_training', True) else samplelib.SampleProcessor.FaceMaskType.NONE
                sample_process_options = samplelib.SampleProcessor.Options( batch_size=self.get_batch_size(), resolution=self.resolution, face_type=self.face_type_enum, mask_type=mask_type_enum, eye_mouth_prio=self.options.get('eyes_prio', False) or self.options.get('mouth_prio', False), augmentation_params=augs, random_warp=False, true_face_power=self.options.get('true_face_power',0.0), random_flip=True )
            except AttributeError as e: io.log_err(f"FATAL: Error accessing SampleProcessor nested class/enum - {e}."); raise
            except Exception as e: io.log_err(f"Error creating SampleProcessor.Options: {e}"); traceback.print_exc(); self.generator_list = None; return
            SampleProcessor_sample_options = sample_process_options; SampleProcessor_sample_dst_options = copy.deepcopy(sample_process_options); SampleProcessor_sample_dst_options.random_flip = self.options.get('random_dst_flip', True); SampleProcessor_sample_options.random_flip = self.options.get('random_src_flip', False)
            # Define output types required by train_step
            output_sample_types_defs = [
                {'sample_type': samplelib.SampleProcessor.SampleType.FACE_IMAGE, 'warp': True,  'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.BGR, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, # warped_src/dst (ct_mode and hsv handled internally by SampleProcessor based on options)
                {'sample_type': samplelib.SampleProcessor.SampleType.FACE_IMAGE, 'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.BGR, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, # target_src/dst
                {'sample_type': samplelib.SampleProcessor.SampleType.FACE_MASK,  'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.G,   'face_mask_type': samplelib.SampleProcessor.FaceMaskType.FULL_FACE, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}, # target_srcm_all/dstm_all
                {'sample_type': samplelib.SampleProcessor.SampleType.FACE_MASK,  'warp': False, 'transform': True, 'channel_type': samplelib.SampleProcessor.ChannelType.G,   'face_mask_type': samplelib.SampleProcessor.FaceMaskType.FULL_FACE_EYES, 'face_type': self.face_type_str, 'data_format': nn.data_format, 'resolution': resolution}  # target_srcm_em/dstm_em
            ]
            generators_count = min(4, multiprocessing.cpu_count());
            if pretrain_enabled:
                 io.log_info(f"Initializing SampleGeneratorFace for PRETRAINING using {pretraining_data_path}")
                 # Pretraining uses the same dataset for src and dst
                 src_dst_generators = [ samplelib.SampleGeneratorFace( pretraining_data_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False) or pretrain_enabled, retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count) ] * generators_count;
                 self.set_training_data_generators([src_dst_generators[0], src_dst_generators[0]])
            else:
                 io.log_info(f"Initializing SampleGeneratorFace for SRC using {training_data_src_path}")
                 src_generators = [ samplelib.SampleGeneratorFace( training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False), retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count ) ] * generators_count;
                 io.log_info(f"Initializing SampleGeneratorFace for DST using {training_data_dst_path}")
                 dst_generators = [ samplelib.SampleGeneratorFace( training_data_dst_path, debug=self.debug, batch_size=self.get_batch_size(), sample_process_options=SampleProcessor_sample_dst_options, output_sample_types=output_sample_types_defs, uniform_yaw_distribution=self.options.get('uniform_yaw', False), retraining_samples=self.options.get('retraining_samples', False), generators_count=generators_count ) ] * generators_count;
                 self.set_training_data_generators([src_generators[0], dst_generators[0]])
        else:
             self.generator_list = None

# --- START OF FILE models/Model_SAEHD/Model.py --- (Part 3/3)

# (Continued from Part 2)
# Inside class Model(tf.keras.Model):

    # --- initialize_sample_generators already included in Part 2 ---
    # def initialize_sample_generators(self): ...

    # --- Helper Methods (Continued) ---

    def get_config_schema_path(self):
        """Returns the path to the JSON schema for model configuration."""
        # Assumes schema file is in the same directory as Model.py
        # This might not be used if config handling is simplified
        schema_path = Path(__file__).parent / "config_schema.json"
        return schema_path if schema_path.exists() else None

    def get_formatted_configuration_path(self):
        """Returns the path to the formatted YAML configuration file."""
        # Assumes yaml file is in the same directory as Model.py
        # This might not be used if config handling is simplified
        format_path = Path(__file__).parent / "formatted_config.yaml"
        return format_path if format_path.exists() else None

    def save_config_file(self, filepath):
        """Saves options to configuration yaml file."""
        if yaml is None: io.log_err("Cannot save config file: PyYAML not installed."); return
        if filepath is None: io.log_err("Cannot save config file: filepath is None."); return

        try:
             # Helper to convert numpy types to standard Python types for YAML
             def convert_type_write(value):
                 if isinstance(value, (np.int32, np.int64)): return int(value.item())
                 elif isinstance(value, (np.float32, np.float64)): return float(value.item())
                 elif isinstance(value, np.bool_): return bool(value.item())
                 elif isinstance(value, (list, tuple)): return [convert_type_write(item) for item in value]
                 elif isinstance(value, dict): return {k_inner: convert_type_write(v_inner) for k_inner, v_inner in value.items()}
                 return value

             # Convert the entire options dictionary
             options_to_save = {k: convert_type_write(v) for k, v in self.options.items()}

             # Dump the converted dictionary to the YAML file
             with open(filepath, 'w') as file:
                 yaml.dump(options_to_save, file, default_flow_style=False, sort_keys=False)
             io.log_info(f"Options saved to {filepath.name}")

        except Exception as e:
            io.log_err(f"Error saving config file to {filepath}: {e}")
            traceback.print_exc()

    # --- Preview Handling ---
    def should_save_preview_history(self):
        """Determines if preview history should be saved based on iteration."""
        # This logic matches the original Trainer.py periodic preview check more closely
        if self.options.get('write_preview_history', False):
             current_iter = self.get_iter()
             # Use a reasonable frequency, maybe every 100 or 500 iterations?
             save_freq = 100
             # Adjust frequency based on resolution or other factors if needed
             # save_freq = 10 * (max(1, self.resolution // 64)) # Original calculation attempt
             return current_iter > 0 and current_iter % save_freq == 0
        return False

    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        """Updates the samples used for generating previews."""
        if not self.is_training: return
        preview_samples_count = self.options.get('preview_samples', 2) # Use updated default
        new_samples = []; new_sample_filenames = []

        # Ensure generators are ready
        if not hasattr(self, 'generator_list') or not self.generator_list or len(self.generator_list) != 2 or self.generator_list[0] is None or self.generator_list[1] is None:
            io.log_info("Warning: Cannot update preview samples: Sample generators not available.")
            return # Cannot update if generators aren't ready

        try:
            # Generate enough samples for the preview batch
            # Need to handle potential batch size differences from main training
            src_gen = self.generator_list[0]
            dst_gen = self.generator_list[1]
            collected_src = []
            collected_dst = []
            collected_src_filenames = []
            collected_dst_filenames = []

            # Collect enough source and destination samples separately
            while len(collected_src) < preview_samples_count:
                src_batch_samples, src_batch_filenames = src_gen.generate_next()
                collected_src.extend(src_batch_samples[1]) # Index 1 is target_src
                collected_src_filenames.extend(src_batch_filenames)
            while len(collected_dst) < preview_samples_count:
                dst_batch_samples, dst_batch_filenames = dst_gen.generate_next()
                collected_dst.extend(dst_batch_samples[1]) # Index 1 is target_dst
                collected_dst_filenames.extend(dst_batch_filenames)

            # Take the required number of samples
            for i in range(preview_samples_count):
                target_s = collected_src[i]
                target_d = collected_dst[i]
                fname_s = collected_src_filenames[i] if i < len(collected_src_filenames) else None
                fname_d = collected_dst_filenames[i] if i < len(collected_dst_filenames) else None
                new_samples.append((target_s, target_d))
                new_sample_filenames.append((fname_s, fname_d))

        except StopIteration: io.log_info("Sample generator exhausted during preview update."); # Should ideally not happen with looping generators
        except Exception as e: io.log_err(f"Error generating samples for preview: {e}"); traceback.print_exc(); return # Stop update on error

        if new_samples:
            self.sample_for_preview = new_samples
            self.sample_filenames = new_sample_filenames # Store associated filenames
            io.log_info(f"Preview samples updated ({len(new_samples)} samples).")
        else:
             io.log_info("Warning: Could not generate any new preview samples.")


    def get_preview_samples(self):
        """Returns the stored samples designated for preview generation."""
        # Check if samples exist and are valid
        if not hasattr(self, 'sample_for_preview') or self.sample_for_preview is None or not isinstance(self.sample_for_preview, list) or len(self.sample_for_preview) == 0:
             io.log_info("Preview samples are None/empty, attempting to update...")
             self.update_sample_for_preview(force_new=True) # Try to generate them
             # Check again after trying to update
             if not hasattr(self, 'sample_for_preview') or self.sample_for_preview is None or not isinstance(self.sample_for_preview, list) or len(self.sample_for_preview) == 0:
                  io.log_err("Failed to update preview samples. Cannot provide preview data.")
                  return None # Return None if still unavailable
        # Return the stored list of (target_src_np, target_dst_np) tuples
        return self.sample_for_preview

    def onGetPreview(self, samples_data):
        """Generates preview images from the provided sample data."""
        if samples_data is None or len(samples_data) == 0: return []
        preview_images = []
        # Use the actual number of samples provided, limited by option
        num_samples_to_show = min(len(samples_data), self.options.get('preview_samples', 2))
        if num_samples_to_show == 0: return []

        batch_src_t = []; batch_dst_t = []
        # Use stored filenames if available, otherwise default to None
        filenames_to_use = getattr(self, 'sample_filenames', [(None, None)] * num_samples_to_show)

        for i in range(num_samples_to_show):
            try: # Add try-except for sample unpacking
                src_sample_np, dst_sample_np = samples_data[i]
                batch_src_t.append(src_sample_np); batch_dst_t.append(dst_sample_np)
            except (ValueError, IndexError) as e:
                io.log_err(f"Error unpacking sample data at index {i}: {e}")
                continue # Skip this sample

        if not batch_src_t or not batch_dst_t: # Check if any valid samples were collected
            io.log_err("No valid samples collected for preview inference.")
            return []

        try: # Convert batch to tensors
            # Use the Keras backend floatx setting
            tensor_dtype = tf.keras.backend.floatx()
            batch_src_t = tf.convert_to_tensor(batch_src_t, dtype=tensor_dtype)
            batch_dst_t = tf.convert_to_tensor(batch_dst_t, dtype=tensor_dtype)
        except Exception as e: io.log_err(f"Error converting preview samples to tensors: {e}"); return []

        try: # Run inference
            # Pass training=True to get all internal paths/outputs needed for different previews
            outputs = self.call([batch_src_t, batch_dst_t], training=True)
            archi_type_final, _ = self._parse_archi()

            # Unpack outputs based on architecture (must match call signature)
            if 'df' in archi_type_final or 'liae' in archi_type_final:
                 p_s_s_b, p_s_sm_b, p_d_d_b, p_d_dm_b, p_s_d_b, p_s_dm_b, _, _ = outputs
            else: io.log_err(f"Unsupported architecture '{archi_type_final}' in onGetPreview."); return []

            # Convert necessary tensors to numpy for image processing
            S_b = batch_src_t.numpy(); D_b = batch_dst_t.numpy();
            # SS_b = p_s_s_b.numpy(); # Src->Src (Not used in final 2x2)
            DD_b = p_d_d_b.numpy(); # Dst->Dst
            SD_b = p_s_d_b.numpy(); # Src->Dst (Prediction)

            # Assemble preview images for each sample
            for i in range(len(S_b)): # Iterate through the actual batch size used for inference
                 S, D, DD, SD = S_b[i], D_b[i], DD_b[i], SD_b[i]
                 
                 # ---- DEBUG Preview Raw Values ----
                 print(f"DEBUG onGetPreview sample {i}: S min/max: {np.min(S):.4f}/{np.max(S):.4f}")
                 print(f"DEBUG onGetPreview sample {i}: D min/max: {np.min(D):.4f}/{np.max(D):.4f}")
                 print(f"DEBUG onGetPreview sample {i}: DD min/max: {np.min(DD):.4f}/{np.max(DD):.4f}")
                 print(f"DEBUG onGetPreview sample {i}: SD (prediction) min/max: {np.min(SD):.4f}/{np.max(SD):.4f}")
                 # ---------------------------------
                 
                 # Normalize & Convert to uint8 BGR
                 S, D, DD, SD = [(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) for img in [S, D, DD, SD]]

                 try: # Assemble 2x2 grid
                     if not (S.shape == D.shape == SD.shape == DD.shape): raise ValueError(f"Mismatched shapes")
                     # Layout: [[S, SD], [D, DD]]
                     top_row = cv2.hconcat([S, SD])
                     bottom_row = cv2.hconcat([D, DD])
                     combined_image = cv2.vconcat([top_row, bottom_row])
                 except Exception as e_concat:
                     io.log_err(f"Error during preview hconcat/vconcat for sample {i}: {e_concat}")
                     # Fallback to just the source image if stacking fails
                     combined_image = S if S is not None else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

                 # --- Title Generation ---
                 title = f"Preview {i}";
                 # Safely access filenames
                 if hasattr(self, 'sample_filenames') and len(self.sample_filenames) > i:
                      sf, df = self.sample_filenames[i];
                      s_name = Path(sf).name if sf else "N/A"
                      d_name = Path(df).name if df else "N/A"
                      # Keep title shorter if needed
                      title += f" (S:{s_name} D:{d_name})"
                 # -----------------------
                 preview_images.append( (title, combined_image) )

        except Exception as e: io.log_err(f"Error during inference or image assembly in onGetPreview: {e}"); traceback.print_exc(); return []
        return preview_images

    def predictor_func(self, face=None, **kwargs):
        """Performs inference for merging."""
        if face is None: raise ValueError("Predictor received None face.");
        # Ensure input is float32, add batch dim
        # Use Keras backend floatx for consistency
        face_t = tf.convert_to_tensor(face[None,...], dtype=tf.keras.backend.floatx());
        # Call model with training=False for inference
        # Output depends on architecture: (predicted_face, src_dst_mask, dst_dst_mask)
        p_f_t, p_s_dm_t, p_d_dm_t = self.call(face_t, training=False);
        # Return face and both masks (src->dst mask, dst->dst mask)
        return np.clip(p_f_t[0].numpy(), 0, 1), np.clip(p_s_dm_t[0].numpy(), 0, 1), np.clip(p_d_dm_t[0].numpy(), 0, 1);

    @staticmethod
    def run_isolated_wscale_tests_static():
        import math # Import needed by the test function
        print("\n===================================")
        print("=== RUNNING ISOLATED WSCALE TESTS (STATIC METHOD - FORWARD PASS ONLY) ===")
        print("===================================")
        
        # Ensure these are available
        StandardKerasConv2D = tf.keras.layers.Conv2D
        StandardKerasDense = tf.keras.layers.Dense
        LeakyReLU_fn = tf.keras.layers.LeakyReLU
        
        # We expect WScaleConv2D and WScaleDense to be imported at the Model.py module level
        # If not, this test will fail at layer instantiation.

        tf.random.set_seed(42)
        batch_size = 4 # Smaller batch for quicker test
        test_input_np = np.random.normal(size=[batch_size, 64, 64, 3]).astype(np.float32)
        test_input = tf.convert_to_tensor(test_input_np)
        
        activation_fn = LeakyReLU_fn(alpha=0.1)

        print(f"\nTest Input shape: {test_input.shape}")
        print(f"Test Input mean: {tf.reduce_mean(test_input).numpy():.6f}")
        print(f"Test Input std: {tf.math.reduce_std(test_input).numpy():.6f}")

        # --- Test WScaleConv2D ---
        print("\n--- WScaleConv2D with default gain=sqrt(2.0) ---")
        try:
            wconv_sqrt2 = WScaleConv2D(filters=64, kernel_size=3, padding='same', name="test_wconv_sqrt2_gain")
            wconv_sqrt2.build(test_input.shape) # Explicit build
            output_logits_sqrt2 = wconv_sqrt2(test_input)
            output_activated_sqrt2 = activation_fn(output_logits_sqrt2)
            if wconv_sqrt2.built:
                print(f"  Kernel Fan-in: {np.prod(wconv_sqrt2.kernel.shape[:-1])}")
                print(f"  Runtime Scale: {wconv_sqrt2.runtime_scale.numpy():.6f}")
            print(f"  Output Logits min/max: {tf.reduce_min(output_logits_sqrt2).numpy():.4f} / {tf.reduce_max(output_logits_sqrt2).numpy():.4f}")
            print(f"  Output Logits mean: {tf.reduce_mean(output_logits_sqrt2).numpy():.6f}")
            print(f"  Output Logits std: {tf.math.reduce_std(output_logits_sqrt2).numpy():.6f}")
            print(f"  Output Activated std: {tf.math.reduce_std(output_activated_sqrt2).numpy():.6f}")
        except Exception as e:
            print(f"  ERROR in WScaleConv2D gain=sqrt(2.0) test: {e}")

        print("\n--- WScaleConv2D with gain=1.0 ---")
        try:
            wconv_gain1 = WScaleConv2D(filters=64, kernel_size=3, gain=1.0, padding='same', name="test_wconv_gain1")
            wconv_gain1.build(test_input.shape)
            output_logits_gain1 = wconv_gain1(test_input)
            output_activated_gain1 = activation_fn(output_logits_gain1)
            if wconv_gain1.built:
                print(f"  Runtime Scale: {wconv_gain1.runtime_scale.numpy():.6f}")
            print(f"  Output Logits std: {tf.math.reduce_std(output_logits_gain1).numpy():.6f}")
            print(f"  Output Activated std: {tf.math.reduce_std(output_activated_gain1).numpy():.6f}")
        except Exception as e:
            print(f"  ERROR in WScaleConv2D gain=1.0 test: {e}")
            
        print("\n--- WScaleConv2D with gain=2.0 ---")
        try:
            wconv_gain2 = WScaleConv2D(filters=64, kernel_size=3, gain=2.0, padding='same', name="test_wconv_gain2")
            wconv_gain2.build(test_input.shape)
            output_logits_gain2 = wconv_gain2(test_input)
            output_activated_gain2 = activation_fn(output_logits_gain2)
            if wconv_gain2.built:
                print(f"  Runtime Scale: {wconv_gain2.runtime_scale.numpy():.6f}")
            print(f"  Output Logits std: {tf.math.reduce_std(output_logits_gain2).numpy():.6f}")
            print(f"  Output Activated std: {tf.math.reduce_std(output_activated_gain2).numpy():.6f}")
        except Exception as e:
            print(f"  ERROR in WScaleConv2D gain=2.0 test: {e}")

        print("\n--- StandardKerasConv2D with He_normal initialization ---")
        try:
            stdconv_he = StandardKerasConv2D(filters=64, kernel_size=3, padding='same', 
                                             kernel_initializer=tf.keras.initializers.he_normal(), name="test_stdconv_he")
            output_std_logits = stdconv_he(test_input)
            output_std_activated = activation_fn(output_std_logits)
            print(f"  Output Logits std: {tf.math.reduce_std(output_std_logits).numpy():.6f}")
            print(f"  Output Activated std: {tf.math.reduce_std(output_std_activated).numpy():.6f}")
        except Exception as e:
            print(f"  ERROR in StandardKerasConv2D He test: {e}")

        print("\n=== Testing WScaleConv2D Chain (5 layers, gain=sqrt(2.0)) ===")
        x = test_input
        current_activation_fn = LeakyReLU_fn(alpha=0.1) # Use separate instance
        for i in range(5):
            try:
                layer_name = f"test_wconv_chain_{i}_sqrt2"
                wconv_chain = WScaleConv2D(filters=64, kernel_size=3, padding='same', name=layer_name) # Default gain = sqrt(2.0)
                # x needs to be built before passing to the layer if its shape might change,
                # but here input shape is fixed for the chain.
                x_logits = wconv_chain(x)
                x = current_activation_fn(x_logits)
                print(f"  Layer {i+1} (name: {layer_name}) Output Logits std: {tf.math.reduce_std(x_logits).numpy():.6f}, Activated std: {tf.math.reduce_std(x).numpy():.6f}")
            except Exception as e:
                print(f"  ERROR in chain layer {i+1}: {e}")
                break
        
        print("\n=== WScaleDense Layer Test (gain=sqrt(2.0)) ===")
        try:
            flattened_input = tf.reshape(test_input, [batch_size, -1])
            print(f"  Flattened Input std: {tf.math.reduce_std(flattened_input).numpy():.6f}")

            wdense_sqrt2 = WScaleDense(units=128, name="test_wdense_sqrt2_gain") # Default gain = sqrt(2.0)
            wdense_sqrt2.build(flattened_input.shape)
            dense_output_logits = wdense_sqrt2(flattened_input)
            dense_output_activated = activation_fn(dense_output_logits)
            if wdense_sqrt2.built:
                print(f"  Kernel Fan-in: {wdense_sqrt2.kernel.shape[0]}")
                print(f"  Runtime Scale: {wdense_sqrt2.runtime_scale.numpy():.6f}")
            print(f"  Output Logits std: {tf.math.reduce_std(dense_output_logits).numpy():.6f}")
            print(f"  Output Activated std: {tf.math.reduce_std(dense_output_activated).numpy():.6f}")
        except Exception as e:
            print(f"  ERROR in WScaleDense gain=sqrt(2.0) test: {e}")

        print("\n=== Verifying WScaleConv2D runtime_scale calculation (gain=sqrt(2.0)) ===")
        try:
            fan_in_calc = 3 * 3 * 3 
            expected_scale_calc = math.sqrt(2.0) / math.sqrt(fan_in_calc)
            print(f"  Expected scale for 3x3x3_in kernel, gain=sqrt(2.0): {expected_scale_calc:.6f}")

            test_wconv_calc = WScaleConv2D(filters=64, kernel_size=3, padding='same', name="test_wconv_calc_scale") # Default gain
            _ = test_wconv_calc(tf.random.normal([1, 32, 32, 3])) 
            actual_scale_calc = test_wconv_calc.runtime_scale.numpy()
            print(f"  Actual scale from WScaleConv2D: {actual_scale_calc:.6f}")
            print(f"  Scales match: {abs(expected_scale_calc - actual_scale_calc) < 1e-6}")
        except Exception as e:
            print(f"  ERROR in runtime_scale verification: {e}")

        print("\n===================================")
        print("=== ISOLATED WSCALE TESTS COMPLETE (FORWARD PASS ONLY) ===")
        print("===================================\n")
        sys.stdout.flush()

    def get_MergerConfig(self):
        """Returns configuration needed for the merger script."""
        # Local import for safety, might need adjustment based on final project structure
        try: import merger
        except ImportError: io.log_err("Merger module not found."); return None, None, None

        archi_type = getattr(self, 'archi_type', self._parse_archi()[0]) # Get archi_type safely
        is_morphable = ('liae' in archi_type)
        # Return predictor function, resolution tuple, and MergerConfigMasked
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=self.face_type_enum, is_morphable=is_morphable)

# --- End of Class Model ---

# --- END OF FILE models/Model_SAEHD/Model.py --- (Part 3/3)