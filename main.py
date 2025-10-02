# --- START OF main.py --- (Aggressive Delayed Imports + Debug Prints + Corrected Globals)

# Minimal top-level imports
import multiprocessing
import os
import sys
import time
import argparse
from pathlib import Path
import traceback # Import traceback for error printing

# --- fixPathAction Class Definition ---
class fixPathAction(argparse.Action):
    """Custom argparse action to expand and normalize paths."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

# --- Main Execution Block ---
if __name__ == "__main__":
    print("START: main execution block")
    # Fix for linux/macOS multiprocessing start method
    if sys.platform != 'win32': # More robust platform check
        try:
            # Check if context is already set, avoid setting twice
            if multiprocessing.get_start_method(allow_none=True) is None:
                 multiprocessing.set_start_method("spawn")
                 print("INFO: Set multiprocessing start method to spawn.")
        except Exception as e: # Catch potential errors like RuntimeError or ValueError
             print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")

    # --- Import TF/Keras FIRST ---
    print("IMPORT: Attempting tensorflow import...")
    try:
        import tensorflow as tf
        # Optionally filter logs here if needed
        import logging
        tf_logger = logging.getLogger('tensorflow'); tf_logger.setLevel(logging.ERROR)
        # Try importing optimizers immediately after TF
        print("IMPORT: Attempting keras.optimizers import...")
        from tensorflow.keras import optimizers
        from tensorflow.keras import backend as K
        print("IMPORT: tensorflow and Keras optimizers imported successfully.")
    except Exception as e:
        print(f"FATAL: Could not import TensorFlow/Keras: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Import DFL Core Components AFTER TF/Keras ---
    print("IMPORT: Attempting DFL core imports (nn, pathex, osex, io)...")
    try:
        # Import nn first - use the SIMPLE version (no internal leras component imports)
        from core.leras import nn
        from core import pathex
        from core import osex
        from core.interact import interact as io

        print("IMPORT: DFL core imported successfully.")
    except Exception as e:
        print(f"FATAL: Could not import DFL core components: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Initialize Main Env (Before nn.initialize) ---
    print("EXEC: nn.initialize_main_env()")
    if hasattr(nn, 'initialize_main_env'):
        try:
             nn.initialize_main_env()
        except Exception as e:
             print(f"ERROR during nn.initialize_main_env: {e}")
             # Decide if this is fatal
    else:
        print("Warning: nn.initialize_main_env not found.")

    # --- Version Check ---
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        print(f"FATAL: Python version {sys.version_info} is not supported. Requires Python 3.6+.")
        sys.exit(1)

    # --- Argument Parser Setup ---
    print("SETUP: Argument parser...")
    exit_code = 0 # Define global exit_code here
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Select command') # Add help for subparsers

    # ========================================================
    # --- process_* function definitions ---
    # ========================================================

    def process_extract(arguments):
        global exit_code # Declare global at the start of function
        # Imports are local to the function
        from core import osex
        from mainscripts import Extractor
        from pathlib import Path
        from core.interact import interact as io # Import io locally

        print("Processing extract...")
        osex.set_process_lowest_prio()

        # Process gpu_idxs argument
        force_gpu_idxs_list = None
        if arguments.force_gpu_idxs is not None:
            try:
                force_gpu_idxs_list = [ int(x.strip()) for x in arguments.force_gpu_idxs.split(',') ]
            except ValueError:
                io.log_err(f"Warning: Invalid format for --force-gpu-idxs '{arguments.force_gpu_idxs}'. Ignoring.") # Use io.log_err
                force_gpu_idxs_list = None

        # Call Extractor main function
        try:
            Extractor.main( detector                = arguments.detector,
                            input_path              = Path(arguments.input_dir),
                            output_path             = Path(arguments.output_dir),
                            output_debug            = arguments.output_debug,
                            manual_fix              = arguments.manual_fix,
                            manual_output_debug_fix = arguments.manual_output_debug_fix,
                            manual_window_size      = arguments.manual_window_size,
                            face_type               = arguments.face_type,
                            max_faces_from_image    = arguments.max_faces_from_image,
                            image_size              = arguments.image_size,
                            jpeg_quality            = arguments.jpeg_quality,
                            cpu_only                = arguments.cpu_only,
                            force_gpu_idxs          = force_gpu_idxs_list,
                          )
        except Exception as e:
             print(f"ERROR during extractor processing: {e}")
             traceback.print_exc()
             exit_code = 1

    def process_sort(arguments):
        global exit_code # Declare global
        from core import osex
        from mainscripts import Sorter
        from pathlib import Path
        from core.interact import interact as io
        print("Processing sort...")
        osex.set_process_lowest_prio()
        try:
            Sorter.main (input_path=Path(arguments.input_dir), sort_by_method=arguments.sort_by_method)
        except Exception as e:
            print(f"ERROR during sort processing: {e}")
            traceback.print_exc()
            exit_code = 1

    def process_util(arguments):
         global exit_code # Declare global
         from core import osex
         from mainscripts import Util
         from pathlib import Path
         from core.interact import interact as io
         from samplelib import PackedFaceset # Import locally needed class

         print("Processing util...")
         osex.set_process_lowest_prio()
         try:
             if arguments.add_landmarks_debug_images: Util.add_landmarks_debug_images (input_path=Path(arguments.input_dir))
             if arguments.recover_original_aligned_filename: Util.recover_original_aligned_filename (input_path=Path(arguments.input_dir))
             if arguments.save_faceset_metadata: Util.save_faceset_metadata_folder (input_path=Path(arguments.input_dir))
             if arguments.restore_faceset_metadata: Util.restore_faceset_metadata_folder (input_path=Path(arguments.input_dir))
             if arguments.pack_faceset: io.log_info ("Performing faceset packing...\r\n"); PackedFaceset.pack( Path(arguments.input_dir), ext=arguments.archive_type)
             if arguments.unpack_faceset: io.log_info ("Performing faceset unpacking...\r\n"); PackedFaceset.unpack( Path(arguments.input_dir) )
         except Exception as e:
              print(f"ERROR during util processing: {e}")
              traceback.print_exc()
              exit_code = 1

    def process_train(arguments):
        global exit_code # Declare global
        # Make imports local if they weren't imported globally
        from core import osex
        from core.leras import nn
        from pathlib import Path
        from core.interact import interact as io

        print("EXEC: process_train started")
        osex.set_process_lowest_prio()

        # Explicit Keras Optimizer Import Attempt (can likely be removed now, but keep for trace)
        print("IMPORT: Re-checking keras.optimizers inside process_train...")
        try:
            from tensorflow.keras import optimizers
            print("IMPORT: keras.optimizers available inside process_train.")
        except Exception as e:
            print(f"ERROR: Could not import Keras optimizers inside process_train (might be acceptable if imported globally): {e}")


        # --- Get args for DeviceConfig ---
        force_gpu_idxs_str = arguments.force_gpu_idxs
        cpu_only_flag = arguments.cpu_only
        processed_gpu_idxs = None
        if force_gpu_idxs_str is not None:
            try:
                processed_gpu_idxs = [int(x.strip()) for x in force_gpu_idxs_str.split(',')]
            except ValueError:
                io.log_err(f"Warning: Invalid format for --force-gpu-idxs '{force_gpu_idxs_str}'. Ignoring.") # Use io.log_err
                processed_gpu_idxs = None

        # --- Create DeviceConfig ---
        if cpu_only_flag:
            device_config = nn.DeviceConfig.CPU()
            io.log_info("Device Config: CPU.")
        elif processed_gpu_idxs is not None:
            device_config = nn.DeviceConfig.GPUIndexes(processed_gpu_idxs)
            io.log_info(f"Device Config: Specific GPUs {processed_gpu_idxs}.")
        else:
            device_config = None # Let nn.initialize handle default
            io.log_info("Device Config: Default GPU.")

        # --- Initialize NN ---
        print("EXEC: nn.initialize()")
        try:
            # Pass floatx/data_format if configurable from args, otherwise defaults in nn.py are used
            nn.initialize(device_config=device_config)
            print("EXEC: nn.initialize() successful.")
        except Exception as e:
             print(f"ERROR: Exception during nn.initialize: {e}")
             traceback.print_exc()
             exit_code = 1
             return # Exit this function on error

        # --- Prepare kwargs for Trainer.main ---
        execute_programs_list = []
        saved_models_path_arg = arguments.model_dir # Get from arguments
        print(f"DEBUG: --model-dir argument from parser: {saved_models_path_arg} (Type: {type(saved_models_path_arg)})")
        if arguments.execute_program:
             for prog_arg in arguments.execute_program:
                  if len(prog_arg) >= 2:
                       try: execute_programs_list.append([int(prog_arg[0]), " ".join(prog_arg[1:])]) # Join remaining parts
                       except ValueError: print(f"Warning: Invalid format for --execute-program {prog_arg}. Skipping.")
                  else: print(f"Warning: Invalid format for --execute-program {prog_arg}. Skipping.")

        # Convert to Path object
        try:
            saved_models_path_obj = Path(saved_models_path_arg)
            print(f"DEBUG: Converted model_dir to Path object: {saved_models_path_obj}")
        except Exception as e:
            print(f"ERROR: Could not create Path object from model_dir: {e}")
            saved_models_path_obj = None # Set to None on error

        # Build the dictionary carefully
        kwargs = {
            'model_class_name'         : arguments.model_name,
            'saved_models_path'        : Path(arguments.model_dir) if arguments.model_dir else None,
            'training_data_src_path'   : Path(arguments.training_data_src_dir) if arguments.training_data_src_dir else None,
            'training_data_dst_path'   : Path(arguments.training_data_dst_dir) if arguments.training_data_dst_dir else None,
            'pretraining_data_path'    : Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
            'pretrained_model_path'    : Path(arguments.pretrained_model_dir) if arguments.pretrained_model_dir is not None else None,
            'no_preview'               : arguments.no_preview,
            'force_model_name'         : arguments.force_model_name,
            'force_gpu_idxs'           : processed_gpu_idxs,
            'cpu_only'                 : cpu_only_flag,
            'silent_start'             : arguments.silent_start,
            'execute_programs'         : execute_programs_list,
            'debug'                    : arguments.debug,
            'saving_time'              : arguments.saving_time,
            'tensorboard_dir'          : arguments.tensorboard_dir,
            'start_tensorboard'        : arguments.start_tensorboard,
            'dump_ckpt'                : arguments.dump_ckpt,
            'flask_preview'            : arguments.flask_preview,
            'config_training_file'     : arguments.config_training_file,
            'auto_gen_config'          : arguments.auto_gen_config,
            'force_gradient_checkpointing': arguments.force_gradient_checkpointing,
            'batch_size'               : arguments.batch_size if arguments.batch_size is not None else 4
        }
        print(f"DEBUG: kwargs['saved_models_path'] being passed to Trainer.main: {kwargs['saved_models_path']}")

        # --- Import and run Trainer ---
        print("IMPORT: Attempting Trainer import...")
        try:
            # Make sure mainscripts is importable
            from mainscripts import Trainer
            print("IMPORT: Trainer imported successfully.")
        except Exception as e:
            print(f"ERROR: Could not import Trainer: {e}")
            traceback.print_exc()
            exit_code = 1
            return # Exit this function on error

        print("EXEC: Calling Trainer.main...")
        try:
            Trainer.main(**kwargs)
        except Exception as e:
            print(f"ERROR during Trainer.main execution: {e}")
            traceback.print_exc()
            exit_code = 1
            # No return here, let main finish

    def process_exportdfm(arguments):
        global exit_code # Declare global
        from core import osex
        from mainscripts import ExportDFM
        from pathlib import Path
        print("Processing exportdfm...")
        osex.set_process_lowest_prio()
        try:
            ExportDFM.main(model_class_name = arguments.model_name, saved_models_path = Path(arguments.model_dir))
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            exit_code = 1

    def process_merge(arguments):
        global exit_code # Declare global
        from core import osex
        from pathlib import Path
        from core.interact import interact as io
        print("Processing merge...")
        osex.set_process_lowest_prio()

        # AVX-512 initialization
        if hasattr(arguments, 'use_avx512') and arguments.use_avx512:
             try:
                 from core.avx512 import initialize
                 avx512_enabled = initialize()
                 if avx512_enabled:
                      io.log_info("AVX-512 optimizations enabled")
                      import sys; import importlib.util;
                      spec = importlib.util.spec_from_file_location("MergeMasked_avx512", str(Path(__file__).parent / 'merger' / 'MergeMasked_avx512.py')); MergeMasked_avx512 = importlib.util.module_from_spec(spec); spec.loader.exec_module(MergeMasked_avx512);
                      import merger.MergeMasked; merger.MergeMasked.MergeMaskedFace = MergeMasked_avx512.MergeMaskedFace; merger.MergeMasked.MergeMasked = MergeMasked_avx512.MergeMasked;
             except ImportError as e:
                 io.log_info(f"Failed to initialize AVX-512 optimizations: {str(e)}")
             except Exception as e:
                 io.log_info(f"Error during AVX-512 setup: {str(e)}")


        from mainscripts import Merger # Import locally

        # Process gpu_idxs argument
        force_gpu_idxs_list = None
        if arguments.force_gpu_idxs is not None:
             try: force_gpu_idxs_list = [ int(x.strip()) for x in arguments.force_gpu_idxs.split(',') ]
             except ValueError: io.log_err(f"Warning: Invalid --force-gpu-idxs. Ignoring."); force_gpu_idxs_list = None

        try:
            Merger.main ( model_class_name       = arguments.model_name,
                          saved_models_path      = Path(arguments.model_dir),
                          force_model_name       = arguments.force_model_name,
                          input_path             = Path(arguments.input_dir),
                          output_path            = Path(arguments.output_dir),
                          output_mask_path       = Path(arguments.output_mask_dir),
                          aligned_path           = Path(arguments.aligned_dir) if arguments.aligned_dir is not None else None,
                          force_gpu_idxs         = force_gpu_idxs_list, # Use processed list
                          cpu_only               = arguments.cpu_only)
        except Exception as e:
            print(f"ERROR during merge processing: {e}")
            traceback.print_exc()
            exit_code = 1

    # --- Define videoed sub-functions (Add 'global exit_code' and try/except) ---
    def process_videoed_extract_video(arguments):
        global exit_code; from core import osex; from mainscripts import VideoEd; osex.set_process_lowest_prio();
        try: VideoEd.extract_video(arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_videoed_cut_video(arguments):
        global exit_code; from core import osex; from mainscripts import VideoEd; osex.set_process_lowest_prio();
        try: VideoEd.cut_video(arguments.input_file, arguments.from_time, arguments.to_time, arguments.audio_track_id, arguments.bitrate)
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_videoed_denoise_image_sequence(arguments):
        global exit_code; from core import osex; from mainscripts import VideoEd; osex.set_process_lowest_prio();
        try: VideoEd.denoise_image_sequence(arguments.input_dir, arguments.factor)
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_videoed_video_from_sequence(arguments):
        global exit_code; from core import osex; from mainscripts import VideoEd; osex.set_process_lowest_prio();
        try: VideoEd.video_from_sequence(input_dir=arguments.input_dir, output_file=arguments.output_file, reference_file=arguments.reference_file, ext=arguments.ext, fps=arguments.fps, bitrate=arguments.bitrate, include_audio=arguments.include_audio, lossless=arguments.lossless)
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;

    # --- Define facesettool sub-functions (Add 'global exit_code' and try/except) ---
    def process_faceset_enhancer(arguments):
        global exit_code; from core import osex; from mainscripts import FacesetEnhancer; from pathlib import Path; osex.set_process_lowest_prio(); force_gpu_idxs_list = None;
        if arguments.force_gpu_idxs is not None:
            try: force_gpu_idxs_list = [ int(x.strip()) for x in arguments.force_gpu_idxs.split(',') ]
            except ValueError: print(f"Warning: Invalid --force-gpu-idxs. Ignoring."); force_gpu_idxs_list = None
        try: FacesetEnhancer.process_folder ( Path(arguments.input_dir), cpu_only=arguments.cpu_only, force_gpu_idxs=force_gpu_idxs_list)
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_faceset_resizer(arguments):
        global exit_code; from core import osex; from mainscripts import FacesetResizer; from pathlib import Path; osex.set_process_lowest_prio();
        try: FacesetResizer.process_folder ( Path(arguments.input_dir) )
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;

    # --- Define dev_test sub-function (Add 'global exit_code' and try/except) ---
    def process_dev_test(arguments):
        global exit_code; from core import osex; from mainscripts import dev_misc; from pathlib import Path; osex.set_process_lowest_prio();
        try: dev_misc.dev_test( arguments.input_dir ) # Pass arguments.input_dir
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;

    # --- Define xseg sub-functions (Add 'global exit_code' and try/except) ---
    def process_xsegeditor(arguments):
        global exit_code; from core import osex; from XSegEditor import XSegEditor; from pathlib import Path; osex.set_process_lowest_prio();
        try: exit_code = XSegEditor.start (Path(arguments.input_dir)); # Assign directly
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_xsegapply(arguments):
        global exit_code; from core import osex; from mainscripts import XSegUtil; from pathlib import Path; osex.set_process_lowest_prio();
        try: XSegUtil.apply_xseg (Path(arguments.input_dir), Path(arguments.model_dir))
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_xsegremove(arguments):
        global exit_code; from core import osex; from mainscripts import XSegUtil; from pathlib import Path; osex.set_process_lowest_prio();
        try: XSegUtil.remove_xseg (Path(arguments.input_dir) )
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_xsegremovelabels(arguments):
        global exit_code; from core import osex; from mainscripts import XSegUtil; from pathlib import Path; osex.set_process_lowest_prio();
        try: XSegUtil.remove_xseg_labels (Path(arguments.input_dir) )
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;
    def process_xsegfetch(arguments):
        global exit_code; from core import osex; from mainscripts import XSegUtil; from pathlib import Path; osex.set_process_lowest_prio();
        try: XSegUtil.fetch_xseg (Path(arguments.input_dir) )
        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); exit_code=1;


    # ========================================================
    # --- Add subparsers ---
    # ========================================================
    print("SETUP: Adding subparsers...")
    # (Add all p = subparsers.add_parser(...) calls for all commands: extract, sort, util, train, etc.)
    # --- Extract ---
    p = subparsers.add_parser( "extract", help="Extract the faces from pictures.") # Corrected help text
    p.add_argument('--detector', dest="detector", choices=['s3fd','manual'], default=None); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir"); p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None); p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None); p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None); p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None); p.add_argument('--image-size', type=int, dest="image_size", default=None); p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None); p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False); p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False); p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368); p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False); p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None); p.set_defaults (func=process_extract)
    # --- Sort ---
    p = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--by', dest="sort_by_method", default=None, choices=("blur", "motion-blur", "face-yaw", "face-pitch", "face-source-rect-size", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "oneface", "final", "final-fast", "absdiff"), help="Method of sorting."); p.set_defaults (func=process_sort)
    # --- Util ---
    p = subparsers.add_parser( "util", help="Utilities.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False); p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False); p.add_argument('--save-faceset-metadata', action="store_true", dest="save_faceset_metadata", default=False); p.add_argument('--restore-faceset-metadata', action="store_true", dest="restore_faceset_metadata", default=False); p.add_argument('--pack-faceset', action="store_true", dest="pack_faceset", default=False); p.add_argument('--unpack-faceset', action="store_true", dest="unpack_faceset", default=False); p.add_argument('--archive-type', dest="archive_type", choices=['zip', 'pak'], default=None); p.set_defaults (func=process_util)
    # --- Train ---
    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir"); p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir"); p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None); p.add_argument('--pretrained-model-dir', action=fixPathAction, dest="pretrained_model_dir", default=None); p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir"); p.add_argument('--model', required=True, dest="model_name", help="Model class name (e.g., SAEHD)."); p.add_argument('--debug', action="store_true", dest="debug", default=False); p.add_argument('--saving-time', type=int, dest="saving_time", default=25); p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False); p.add_argument('--force-model-name', dest="force_model_name", default=None); p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False); p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None); p.add_argument('--silent-start', action="store_true", dest="silent_start", default=False); p.add_argument('--tensorboard-logdir', action=fixPathAction, dest="tensorboard_dir"); p.add_argument('--start-tensorboard', action="store_true", dest="start_tensorboard", default=False); p.add_argument('--config-training-file', action=fixPathAction, dest="config_training_file"); p.add_argument('--auto-gen-config', action="store_true", dest="auto_gen_config", default=False); p.add_argument('--force-gradient-checkpointing', action="store_true", dest="force_gradient_checkpointing", default=False); p.add_argument('--dump-ckpt', action="store_true", dest="dump_ckpt", default=False); p.add_argument('--flask-preview', action="store_true", dest="flask_preview", default=False); p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+'); p.add_argument('--batch-size', type=int, dest="batch_size", default=None, help="Override batch size per replica."); p.set_defaults (func=process_train)
    # --- ExportDFM ---
    p = subparsers.add_parser( "exportdfm", help="Export model to use in DeepFaceLive."); p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir"); p.add_argument('--model', required=True, dest="model_name", help="Model class name."); p.set_defaults (func=process_exportdfm)
    # --- Merge ---
    p = subparsers.add_parser( "merge", help="Merger"); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir"); p.add_argument('--output-mask-dir', required=True, action=fixPathAction, dest="output_mask_dir"); p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", default=None); p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir"); p.add_argument('--model', required=True, dest="model_name", help="Model class name."); p.add_argument('--force-model-name', dest="force_model_name", default=None); p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False); p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None); p.add_argument('--use-avx512', action="store_true", dest="use_avx512", default=False); p.set_defaults(func=process_merge)
    # --- VideoEd ---
    videoed_parser = subparsers.add_parser( "videoed", help="Video processing.").add_subparsers(); p = videoed_parser.add_parser( "extract-video", help="Extract images from video file."); p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file"); p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir"); p.add_argument('--output-ext', dest="output_ext", default=None); p.add_argument('--fps', type=int, dest="fps", default=None); p.set_defaults(func=process_videoed_extract_video); p = videoed_parser.add_parser( "cut-video", help="Cut video file."); p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file"); p.add_argument('--from-time', dest="from_time", default=None); p.add_argument('--to-time', dest="to_time", default=None); p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None); p.add_argument('--bitrate', type=int, dest="bitrate", default=None); p.set_defaults(func=process_videoed_cut_video); p = videoed_parser.add_parser( "denoise-image-sequence", help="Denoise sequence of images."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--factor', type=int, dest="factor", default=None); p.set_defaults(func=process_videoed_denoise_image_sequence); p = videoed_parser.add_parser( "video-from-sequence", help="Make video from image sequence."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file"); p.add_argument('--reference-file', action=fixPathAction, dest="reference_file"); p.add_argument('--ext', dest="ext", default='png'); p.add_argument('--fps', type=int, dest="fps", default=None); p.add_argument('--bitrate', type=int, dest="bitrate", default=None); p.add_argument('--include-audio', action="store_true", dest="include_audio", default=False); p.add_argument('--lossless', action="store_true", dest="lossless", default=False); p.set_defaults(func=process_videoed_video_from_sequence);
    # --- FacesetTool ---
    facesettool_parser = subparsers.add_parser( "facesettool", help="Faceset tools.").add_subparsers(); p = facesettool_parser.add_parser ("enhance", help="Enhance details in DFL faceset."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False); p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None); p.set_defaults(func=process_faceset_enhancer); p = facesettool_parser.add_parser ("resize", help="Resize DFL faceset."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults(func=process_faceset_resizer);
    # --- Dev Test ---
    p = subparsers.add_parser( "dev_test", help=""); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults (func=process_dev_test);
    # --- XSeg ---
    xseg_parser = subparsers.add_parser( "xseg", help="XSeg tools.").add_subparsers(); p = xseg_parser.add_parser( "editor", help="XSeg editor."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults (func=process_xsegeditor); p = xseg_parser.add_parser( "apply", help="Apply trained XSeg model."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir"); p.set_defaults (func=process_xsegapply); p = xseg_parser.add_parser( "remove", help="Remove applied XSeg masks."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults (func=process_xsegremove); p = xseg_parser.add_parser( "remove_labels", help="Remove XSeg labels."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults (func=process_xsegremovelabels); p = xseg_parser.add_parser( "fetch", help="Copies faces containing XSeg polygons."); p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir"); p.set_defaults (func=process_xsegfetch);


    # --- Default / Help ---
    def bad_args(arguments):
        parser.print_help()
        # Use exit code 1 for bad arguments to indicate an error
        exit(1)
    parser.set_defaults(func=bad_args)

    # --- Parse Arguments and Execute ---
    print("EXEC: Parsing arguments...")
    arguments = parser.parse_args()
    print(f"EXEC: Arguments parsed. Calling function: {arguments.func.__name__ if hasattr(arguments, 'func') else 'None'}")

    # Check if a function was assigned (i.e., a valid subcommand was given)
    if hasattr(arguments, 'func'):
        try:
            arguments.func(arguments)
        except Exception as e:
            print(f"FATAL: Error during execution of {arguments.func.__name__}: {e}")
            traceback.print_exc()
            exit_code = 1 # Set exit code on error
    else:
        # If no subcommand, func won't be set, call bad_args to show help
        bad_args(arguments)


    # --- Final Exit ---
    if exit_code == 0:
        print ("Done.")
    else:
        print("Execution finished with errors.")
    # Exit with the determined exit code
    exit(exit_code)

# --- END OF main.py ---