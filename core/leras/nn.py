# --- START OF FILE core/leras/nn.py --- (Final Simplified Version)
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import numpy as np
from core.interact import interact as io
from .device import Devices # Import Devices class

# --- Import TF at the TOP ---
# This ensures TensorFlow and its Keras API are loaded early.
try:
    import tensorflow as tf
except ImportError as e:
    print("FATAL: TensorFlow import failed. Please ensure it is installed correctly for your environment.")
    print(f"Error details: {e}")
    sys.exit(1)

# Set up logging to suppress verbose TensorFlow/Werkzeug messages if desired
import logging
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR) # Suppress INFO and WARNING from TF

class nn():
    # --- Static Class Variables ---
    # Assign static reference to the imported TF module
    tf = tf

    # Core configuration state, managed by static methods
    current_DeviceConfig = None
    tf_default_device_name = None
    data_format = "NHWC" # Default, will be changed by initialize/set_data_format
    conv2d_ch_axis = 3   # Updated by set_data_format
    conv2d_spatial_axes = [1,2] # Updated by set_data_format
    floatx = None # Set by initialize based on Keras backend

    # NO component placeholders or import helper needed in this version

    @staticmethod
    def initialize_main_env():
        """Initializes the main environment (CUDA paths etc.) via Devices class."""
        # This should be called very early, before any TF imports if possible,
        # but placing it here allows access via nn.initialize_main_env()
        Devices.initialize_main_env()

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NHWC"):
        """
        Initializes TensorFlow environment: sets visible devices, memory growth,
        global precision policy, and data format.
        Should be called ONCE early in the application startup (e.g., in main.py's process_train).
        """
        if nn.tf is None:
            # This check should ideally never fail if the top-level import worked
            raise Exception("Tensorflow module (nn.tf) is not available.")

        if device_config is None:
            # If no specific config passed, get the default (e.g., BestGPU or user choice)
            device_config = nn.getCurrentDeviceConfig()
        # Store the determined config for reference
        nn.setCurrentDeviceConfig(device_config)

        # --- Configure Visible Devices ---
        tf_gpus = []
        try:
            # List physical GPUs recognized by TensorFlow
            tf_gpus = tf.config.experimental.list_physical_devices('GPU')
        except Exception as e:
            # Log warning if listing devices fails (e.g., driver issues)
            print(f"Warning: Could not list physical GPU devices: {e}")

        visible_gpus = []
        if len(device_config.devices) > 0 and tf_gpus:
            # If user specified GPUs and TF found some GPUs...
            allowed_indices = [dev.index for dev in device_config.devices]
            print("Configuring specified GPUs.")
            for gpu in tf_gpus:
                try:
                    # Extract index from TF device name (e.g., '/physical_device:GPU:0')
                    gpu_index = int(gpu.name.split(':')[-1])
                    if gpu_index in allowed_indices:
                        visible_gpus.append(gpu)
                        # Enable memory growth to prevent TF allocating all VRAM at once
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"GPU {gpu_index} ({gpu.name}) memory growth set to True.")
                except Exception as e:
                    # Catch potential errors during parsing or setting memory growth
                    print(f"Could not parse or configure GPU: {gpu.name}. Error: {e}")

            if not visible_gpus:
                # Fallback to CPU if selected GPUs are not found/available
                print("Warning: Specified GPUs not found or available. Falling back to CPU.")
                tf.config.set_visible_devices([], 'GPU')
                nn.tf_default_device_name = '/CPU:0'
            else:
                # Set TF to only see the selected GPUs
                tf.config.set_visible_devices(visible_gpus, 'GPU')
                print(f"Visible GPUs set to: {[gpu.name for gpu in visible_gpus]}")
                # Set default device name based on the first visible GPU's TF name
                try:
                    nn.tf_default_device_name = f'/GPU:{visible_gpus[0].name.split(":")[-1]}'
                except Exception: # Fallback if name parsing fails
                     nn.tf_default_device_name = '/GPU:0' # Default TF GPU name
        else:
            # Explicit CPU configuration requested or no GPUs found by TF
            print("Using CPU only." if not tf_gpus else "No specific GPUs selected, using CPU.")
            tf.config.set_visible_devices([], 'GPU')
            nn.tf_default_device_name = '/CPU:0'


        # --- Set Float Precision via tf.keras Backend ---
        try:
            if floatx == "float32":
                tf.keras.backend.set_floatx('float32')
                nn.floatx = tf.float32
            elif floatx == "float16":
                # Check if mixed precision API exists in the loaded TF/Keras version
                if hasattr(tf.keras, 'mixed_precision') and hasattr(tf.keras.mixed_precision, 'Policy'):
                    tf.keras.backend.set_floatx('float16')
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Mixed precision policy set to 'mixed_float16'")
                    nn.floatx = tf.float16
                else:
                    # Fallback if mixed_precision module/Policy is missing
                    print("Warning: tf.keras.mixed_precision not available or incomplete in this TF version. Using float32.")
                    tf.keras.backend.set_floatx('float32')
                    nn.floatx = tf.float32
            else:
                raise ValueError(f"unsupported floatx {floatx}")
        except Exception as e:
             print(f"Warning: Could not set Keras floatx policy: {e}")
             # Basic fallback if Keras backend fails
             nn.floatx = tf.float32 if floatx=="float32" else tf.float16


        # --- Set Data Format ---
        nn.set_data_format(data_format)
        # Log final settings using tf.keras.backend if possible
        try:
            current_floatx = tf.keras.backend.floatx()
            current_data_format = tf.keras.backend.image_data_format()
            io.log_info(f"Leras nn initialized. floatx={current_floatx}, data_format={nn.data_format} (Keras backend: {current_data_format})")
        except Exception:
            # Fallback logging if Keras backend failed
             io.log_info(f"Leras nn initialized. floatx={str(nn.floatx)}, data_format={nn.data_format}")


    @staticmethod
    def set_data_format(data_format):
         """Sets the data format ('NHWC' or 'NCHW') for leras and Keras backend."""
         if data_format not in ["NHWC", "NCHW"]:
             raise ValueError(f"unsupported data_format {data_format}")
         nn.data_format = data_format
         # Keras backend expects lowercase 'channels_last' or 'channels_first'
         keras_data_format = 'channels_last' if data_format == "NHWC" else 'channels_first'
         try:
             # Use tf.keras directly
             tf.keras.backend.set_image_data_format(keras_data_format)
         except Exception as e:
             print(f"Warning: Failed to set Keras image data format: {e}")

         # Update internal axes variables used by some leras components
         if data_format == "NHWC":
             nn.conv2d_ch_axis = 3
             nn.conv2d_spatial_axes = [1,2]
         elif data_format == "NCHW":
             nn.conv2d_ch_axis = 1
             nn.conv2d_spatial_axes = [2,3]

    @staticmethod
    def getCurrentDeviceConfig():
        """Gets the current device configuration."""
        if nn.current_DeviceConfig is None:
            # Default to BestGPU if not set previously
            nn.current_DeviceConfig = nn.DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
         """Sets the current device configuration."""
         # Ensure it's a DeviceConfig instance (using the inner class)
         if not isinstance(device_config, nn.DeviceConfig):
              raise TypeError("device_config must be an instance of nn.DeviceConfig")
         nn.current_DeviceConfig = device_config

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        """Asks user to choose GPU indices interactively."""
        # (Implementation from previous versions - ensures user interaction works)
        devices = Devices.getDevices()
        if len(devices) == 0 and allow_cpu:
            io.log_info("No GPUs detected. Using CPU.")
            return []
        elif len(devices) == 0 and not allow_cpu:
            io.log_err("No GPUs detected and CPU not allowed.")
            return None

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        best_device_indexes_list = []
        if suggest_all_gpu:
            best_device_indexes_list = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device = devices.get_best_device()
            if best_device:
                 best_device_indexes_list = [dev.index for dev in devices.get_equal_devices(best_device)]
            else: # Fallback
                 best_device_indexes_list = all_devices_indexes
        else: # Suggest best single GPU
            best_device = devices.get_best_device()
            if best_device:
                best_device_indexes_list = [ best_device.index ]
            elif allow_cpu: # Suggest CPU if no GPU found/best
                best_device_indexes_list = []
            else: # No best GPU, CPU not allowed
                io.log_err("Could not determine best GPU.")
                return None # Return None indicates error

        # Format suggestion string
        if not best_device_indexes_list and allow_cpu:
             best_device_indexes_str = "CPU"
        elif best_device_indexes_list:
             best_device_indexes_str = ",".join([str(x) for x in best_device_indexes_list])
        else: # No GPUs suggested, CPU not allowed
            best_device_indexes_str = ""

        # --- Prompt User ---
        io.log_info ("")
        if choose_only_one:
            io.log_info ("Choose one GPU idx.")
        else:
            io.log_info ("Choose one or several GPU idxs (separated by comma).")
        io.log_info ("")
        if allow_cpu:
            io.log_info ("[CPU] : CPU")
        for device in devices:
            io.log_info (f"  [{device.index}] : {device.name}")
        io.log_info ("")

        # --- Input Loop ---
        choosed_idxs = [] # Initialize outside loop
        while True:
            try:
                prompt = f"Which {'GPU index' if choose_only_one else 'GPU indexes'} to choose?"
                # Use show_default_value=True if io.input_str supports it
                choosed_idxs_str = io.input_str(prompt, best_device_indexes_str, show_default_value=(best_device_indexes_str != ""))

                if allow_cpu and choosed_idxs_str.strip().lower() == "cpu":
                    choosed_idxs = []
                    break # Valid choice: CPU

                if not choosed_idxs_str: # Handle empty input
                    if allow_cpu and best_device_indexes_str == "CPU":
                        choosed_idxs = []
                        break # Accept default CPU
                    elif best_device_indexes_list:
                         choosed_idxs = best_device_indexes_list
                         print(f"Using default: {best_device_indexes_str}")
                         break # Accept default GPU(s)
                    else:
                        print("Invalid input. Please select device(s).")
                        continue

                # Parse indices
                parsed_idxs = [ int(x.strip()) for x in choosed_idxs_str.split(',') if x.strip().isdigit() ]

                # Validate indices
                valid_selection = True
                choosed_idxs = [] # Reset for validation

                if choosed_idxs_str and not parsed_idxs: # Check if input was non-empty but parsing failed
                    print("Invalid input format. Please enter numbers separated by commas.")
                    valid_selection = False
                    # Fall through to the 'if not valid_selection: continue' below

                if valid_selection and parsed_idxs: # Only validate indices if parsing was successful
                    for idx in parsed_idxs:
                        if idx not in all_devices_indexes:
                            print(f"Error: GPU index {idx} is not available.")
                            valid_selection = False
                            break # Exit inner loop (index check)
                        choosed_idxs.append(idx) # Add valid index to final list

                if not valid_selection:
                    continue # Restart outer loop if any validation failed

                # Check number of indices based on choose_only_one (use validated choosed_idxs)
                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break # Exit loop for valid single GPU choice
                    # Handle cases where only invalid chars were entered OR too many GPUs selected
                    elif len(choosed_idxs) == 0 and not allow_cpu:
                         print("Invalid input. Please enter a valid GPU index.")
                    elif len(choosed_idxs) > 1:
                        print("Please choose only one GPU index.")
                    else: # Includes len==0 with allow_cpu=True after invalid chars
                         print("Invalid input.")
                    continue # Restart loop
                else: # Multi-GPU selection
                    if len(choosed_idxs) >= 1:
                         break # Exit loop for valid multi-GPU choice
                    else: # Should only happen if initial input was invalid and parsed_idxs was empty
                         print("Invalid input. Please enter at least one valid GPU index or 'CPU'.")
                         continue

            except ValueError:
                print("Invalid input. Please enter numbers separated by commas, or 'CPU'.")
            except Exception as e:
                print(f"An unexpected error occurred during device selection: {e}")
                return None # Exit on unexpected error

        io.log_info ("")
        return choosed_idxs # Return the final validated list


    # Keep DeviceConfig subclass definition (make sure syntax is correct)
    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            """Factory method to create DeviceConfig based on user input."""
            idxs = nn.ask_choose_device_idxs(*args,**kwargs)
            # Default to CPU config if error or no selection returns None
            if idxs is None:
                 return nn.DeviceConfig.CPU()
            return nn.DeviceConfig.GPUIndexes(idxs)

        def __init__ (self, devices=None):
            """Initializes DeviceConfig."""
            devices = devices or []
            # Ensure 'devices' is an instance of the Devices class
            if not isinstance(devices, Devices):
                devices = Devices(devices)
            self.devices = devices
            self.cpu_only = len(self.devices) == 0

        @staticmethod
        def BestGPU():
            """Creates DeviceConfig using the best available GPU."""
            devices = Devices.getDevices()
            if not devices: return nn.DeviceConfig.CPU()
            best_device = devices.get_best_device()
            return nn.DeviceConfig([best_device]) if best_device else nn.DeviceConfig.CPU()

        @staticmethod
        def WorstGPU():
            """Creates DeviceConfig using the worst available GPU."""
            devices = Devices.getDevices()
            if not devices: return nn.DeviceConfig.CPU()
            worst_device = devices.get_worst_device()
            return nn.DeviceConfig([worst_device]) if worst_device else nn.DeviceConfig.CPU()

        @staticmethod
        def GPUIndexes(indexes):
            """Creates DeviceConfig using specific GPU indices."""
            if not indexes: # Use CPU if list is empty
                return nn.DeviceConfig.CPU()
            # Get devices based on indices
            all_devices = Devices.getDevices()
            if not all_devices: # Handle case where no devices detected at all
                 print("Warning: No devices found by Devices class.")
                 return nn.DeviceConfig.CPU()
            selected_devices = all_devices.get_devices_from_index_list(indexes)
            # Return CPU config if specified indices resulted in no valid devices
            return nn.DeviceConfig(selected_devices) if selected_devices else nn.DeviceConfig.CPU()

        @staticmethod
        def CPU():
            """Creates DeviceConfig forcing CPU usage."""
            return nn.DeviceConfig([]) # Pass empty list for CPU config

# --- END OF FILE core/leras/nn.py ---