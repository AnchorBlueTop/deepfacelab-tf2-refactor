# --- START OF mainscripts/Trainer.py ---

import os
import sys
import traceback
# import queue # No longer needed for basic loop
# import threading # No longer needed for basic loop
import time
import cProfile
import pstats
from enum import Enum

import numpy as np
import itertools
from pathlib import Path
import cv2
import tensorflow as tf # Import TF2

# Import necessary DFL components (ensure paths are correct)
from core import pathex
from core import imagelib
import models # Imports the __init__.py which imports model classes
from models.Model_SAEHD import Model as SAEHDModel # Your existing model import
from core.interact import interact as io
from core.leras import nn # Import refactored nn
import logging # Keep for TF/Werkzeug log filtering if needed
import datetime # Keep for TensorBoard log dir naming

# Import the Callback base class
from tensorflow.keras.callbacks import Callback

# --- TensorBoard Helpers (Keep or adapt for tf.summary) ---
# Example: TensorBoardTool class might still be useful for launching server
class TensorBoardTool:
    def __init__(self, dir_path): self.dir_path = dir_path
    def run(self):
        # (Keep the original TensorBoard launch logic if needed separately)
        # Note: Integrated TensorBoard logging uses tf.summary below
        pass

# --- Preview Helpers ---
# Zoom Enum and preview functions can be kept
class Zoom(Enum):
    ZOOM_25 = (1 / 4, '25%'); ZOOM_33 = (1 / 3, '33%'); ZOOM_50 = (1 / 2, '50%'); ZOOM_67 = (2 / 3, '67%'); ZOOM_75 = (3 / 4, '75%'); ZOOM_80 = (4 / 5, '80%'); ZOOM_90 = (9 / 10, '90%'); ZOOM_100 = (1, '100%'); ZOOM_110 = (11 / 10, '110%'); ZOOM_125 = (5 / 4, '125%'); ZOOM_150 = (3 / 2, '150%'); ZOOM_175 = (7 / 4, '175%'); ZOOM_200 = (2, '200%'); ZOOM_250 = (5 / 2, '250%'); ZOOM_300 = (3, '300%'); ZOOM_400 = (4, '400%'); ZOOM_500 = (5, '500%')
    def __init__(self, scale, label): self.scale = scale; self.label = label
    def prev(self): cls = self.__class__; members = list(cls); index = members.index(self) - 1; return members[index] if index >= 0 else self
    def next(self): cls = self.__class__; members = list(cls); index = members.index(self) + 1; return members[index] if index < len(members) else self

def scale_previews(previews, zoom=Zoom.ZOOM_100):
    scaled = []; scale_factor = zoom.scale
    for pn, prgb in previews:
        if scale_factor == 1: scaled.append((pn, prgb))
        elif scale_factor < 1: scaled.append((pn, cv2.resize(prgb, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)))
        else: scaled.append((pn, cv2.resize(prgb, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)));
    return scaled

def create_preview_pane_image(previews, selected_preview, loss_history,
                              show_last_history_iters_count, iteration, batch_size, zoom=Zoom.ZOOM_100):
    if not previews: return np.zeros((100, 100, 3), dtype=np.uint8) # Handle empty preview case

    scaled_previews = scale_previews(previews, zoom)
    selected_preview = selected_preview % len(scaled_previews) # Ensure valid index
    selected_preview_name = scaled_previews[selected_preview][0]
    # Ensure preview data is uint8 BGR for display
    selected_preview_bgr = scaled_previews[selected_preview][1]
    if selected_preview_bgr.dtype != np.uint8:
         selected_preview_bgr = (np.clip(selected_preview_bgr, 0, 1) * 255).astype(np.uint8)

    h, w, c = selected_preview_bgr.shape

    # HEAD
    head_lines = [
        '[S]:Save [B]:Backup [Enter]:Exit [-/+]:Zoom: %s' % zoom.label,
        '[P]:Update Preview [Space]:Next Preview', # Removed [L] for history range
        'Preview: "%s" [%d/%d] Iter: %d' % (selected_preview_name, selected_preview + 1, len(previews), iteration)
    ]
    head_line_height = int(15 * zoom.scale); head_height = len(head_lines) * head_line_height;
    # Create head with same dtype as preview image
    head = np.ones((head_height, w, c), dtype=selected_preview_bgr.dtype) * 25 # Dark gray background (0.1 * 255)

    # Draw text (assuming imagelib.get_text_image returns float 0-1)
    for i in range(0, len(head_lines)):
        t = i * head_line_height; b = (i + 1) * head_line_height;
        text_img_float = imagelib.get_text_image((head_line_height, w, c), head_lines[i], color=[0.8]*c)
        # Add text image (scaled to 0-255) to head
        head[t:b, 0:w] = np.clip(head[t:b, 0:w] + (text_img_float * (255*0.8)).astype(head.dtype), 0, 255)


    final_list = [head]

    # Loss history plotting needs reimplementation using matplotlib or similar
    # Skipping for now to focus on core training loop
    # if loss_history is not None and len(loss_history) > 0:
    #     if show_last_history_iters_count == 0: loss_history_to_show = loss_history
    #     else: loss_history_to_show = loss_history[-show_last_history_iters_count:]
    #     lh_height = int(100 * zoom.scale)
    #     # Requires refactored get_loss_history_preview or replacement
    #     # lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iteration, w, c, lh_height)
    #     # TEMP: Placeholder for loss history area
    #     lh_img = np.ones((lh_height, w, c), dtype=selected_preview_bgr.dtype) * 10
    #     final_list.append(lh_img)

    final_list.append(selected_preview_bgr) # Add the preview image
    final = cv2.vconcat(final_list) # Use vconcat for efficiency
    # final = np.clip(final, 0, 255) # Should already be clipped/correct dtype
    return final

# --- LR Dropout Callback ---
class LRDropoutCallback(Callback):
    """
    Applies learning rate dropout by potentially skipping optimizer steps.
    Must be called manually at the start of each training batch.
    """
    def __init__(self, lr_dropout_rate):
        super().__init__()
        if not 0.0 <= lr_dropout_rate <= 1.0:
            raise ValueError("lr_dropout_rate must be between 0.0 and 1.0")
        self.lr_dropout_rate = float(lr_dropout_rate) # Probability of KEEPING update
        self.apply_gradients = tf.Variable(True, dtype=tf.bool, trainable=False) # Use TF variable

    # Call this manually before strategy.run
    # @tf.function # Decorate if complex logic is added later
    def on_train_batch_begin(self, batch, logs=None):
        # Decide whether to apply gradients for this batch
        if self.lr_dropout_rate < 1.0:
            apply_update = tf.random.uniform(shape=()) < self.lr_dropout_rate
            self.apply_gradients.assign(apply_update)
        else:
            self.apply_gradients.assign(True)

# --- Main Function ---
def main(**kwargs):
    io.log_info("Running trainer (TF2 Refactored).\r\n")
    global _train_summary_writer # Use global summary writer variable

    # --- Extract Args ---
    model_class_name = kwargs.get('model_class_name')
    saved_models_path = Path(kwargs.get('saved_models_path'))
    print(f"DEBUG: Trainer.main received saved_models_path: {saved_models_path} (Type: {type(saved_models_path)})")
    training_data_src_path = Path(kwargs.get('training_data_src_path'))
    training_data_dst_path = Path(kwargs.get('training_data_dst_path'))
    pretraining_data_path = Path(kwargs.get('pretraining_data_path')) if kwargs.get('pretraining_data_path') else None
    # pretrained_model_path = Path(kwargs.get('pretrained_model_path')) if kwargs.get('pretrained_model_path') else None # Handled by checkpoint
    no_preview = kwargs.get('no_preview', False)
    force_model_name = kwargs.get('force_model_name')
    force_gpu_idxs = kwargs.get('force_gpu_idxs') # Used by initialize if needed
    cpu_only = kwargs.get('cpu_only', False) # Used by initialize if needed
    silent_start = kwargs.get('silent_start', False)
    execute_programs = kwargs.get('execute_programs', [])
    debug = kwargs.get('debug', False)
    tensorboard_dir = kwargs.get('tensorboard_dir')
    start_tensorboard = kwargs.get('start_tensorboard', False) # For external launch tool
    config_training_file = kwargs.get('config_training_file')
    save_interval_min = kwargs.get('saving_time', 25)
    # force_gradient_checkpointing passed to model init

    # --- Basic Path Checks ---
    if not training_data_src_path.exists(): training_data_src_path.mkdir(exist_ok=True, parents=True)
    if not training_data_dst_path.exists(): training_data_dst_path.mkdir(exist_ok=True, parents=True)
    if not saved_models_path.exists(): saved_models_path.mkdir(exist_ok=True, parents=True)

    try:
        # --- Initialize TF and Devices (assuming called previously in main.py) ---
        # nn.initialize_mp(suppress_debug_print=True) # Removed - should be called in main.py before Trainer.main
        # --- Setup Distribution Strategy ---
        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        io.log_info(f'Number of devices: {num_replicas}')

        # Batch size calculation
        per_replica_batch_size = kwargs.get('batch_size', 4) # This is what model init expects
        global_batch_size = per_replica_batch_size * num_replicas
        io.log_info(f'Per-Replica Batch Size: {per_replica_batch_size}')
        io.log_info(f'Global Batch Size: {global_batch_size}')

        # --- Instantiate Model and Optimizers within Strategy Scope ---
        with strategy.scope():
            # ---- RUN ISOLATED WSCALE TESTS ONCE (NOW INSIDE SCOPE) ----
            print("Trainer.py: About to run isolated WScale tests (inside strategy scope)...")
            SAEHDModel.run_isolated_wscale_tests_static() 
            print("Trainer.py: Isolated WScale tests finished.")
            # sys.exit("Exiting after WScale layer tests.") # Uncomment to stop after tests
            # ---------------------------------------------------------
            
            # Pass per-replica batch size to model if it needs it for internal calcs
            model = models.import_model(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        silent_start=silent_start,
                        config_training_file=config_training_file,
                        auto_gen_config=kwargs.get("auto_gen_config", False),
                        force_gradient_checkpointing=kwargs.get("force_gradient_checkpointing", False),
                        debug=debug,
                        batch_size=per_replica_batch_size # Pass per-replica size
                        # **kwargs passed from main call can go here if needed by model init
                        )
            # Optimizers are created inside model init

            # LR Dropout Callback (commented out to simplify control flow)
            # lr_dropout_callback = None
            # if model.is_training and hasattr(model, 'optimizer_G') and model.optimizer_G is not None:
            #     lr_dropout_rate = getattr(model, 'lr_dropout_rate', 1.0) # Get rate from model
            #     if lr_dropout_rate < 1.0:
            #          # Pass the rate only, optimizer reference set internally? No, pass optimizer.
            #          lr_dropout_callback = LRDropoutCallback(lr_dropout_rate) # Pass rate only
            #          io.log_info(f"LRDropoutCallback initialized with rate {lr_dropout_rate}.")
            lr_dropout_callback = None # Keep variable defined but unused


        # --- Checkpoint Management ---
        checkpointables = {'optimizer_G': model.optimizer_G}
        if hasattr(model, 'optimizer_D_gan') and model.optimizer_D_gan: checkpointables['optimizer_D_gan'] = model.optimizer_D_gan
        if hasattr(model, 'optimizer_D_code') and model.optimizer_D_code: checkpointables['optimizer_D_code'] = model.optimizer_D_code
        # Use the model itself for checkpointing its layers/variables
        checkpointables['model'] = model
        # if hasattr(model, 'encoder'): checkpointables['encoder'] = model.encoder # etc. - checkpointing 'model' is simpler

        checkpoint = tf.train.Checkpoint(**checkpointables)
        model_save_path = model.get_strpath_storage_for_file('') # Directory path
        # Keep only the latest checkpoint to simulate overwriting
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_save_path, max_to_keep=1)
        io.log_info(f"Checkpoint manager configured to keep only the latest checkpoint (max_to_keep=1).") # Add log

        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        if checkpoint_manager.latest_checkpoint:
            io.log_info(f"Restored from {checkpoint_manager.latest_checkpoint}")
            # status.assert_consumed() # Use this for stricter checks after full setup
            status.expect_partial() # Allow partial restores during refactoring
        else:
            io.log_info("Initializing model from scratch.")
            model.set_iter(0) # Ensure iter starts at 0 if no checkpoint

        # --- TensorBoard Setup ---
        summary_writer = None
        if tensorboard_dir is not None:
            log_dir = Path(tensorboard_dir) / model.get_model_name() / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir.mkdir(parents=True, exist_ok=True)
            # Create writer within strategy scope if it writes variables? Usually not needed.
            summary_writer = tf.summary.create_file_writer(str(log_dir))
            _train_summary_writer = summary_writer # Assign to global for potential external access? Risky.
            io.log_info(f"TensorBoard logs saving to: {log_dir}")
        else:
            summary_writer = tf.summary.create_noop_writer()
            _train_summary_writer = summary_writer # Assign noop writer

        # --- Prepare Training Data ---
        # Needs correct data types and shapes based on SampleGenerator output
        # Assuming SampleGenerator yields numpy arrays
        # --- Redefined sample_generator_func ---
        def sample_generator_func():
             # Get the initialized generators from the model
             gen_list = model.generator_list # Access the generators
             if gen_list is None or len(gen_list) != 2:
                  raise Exception("Model generators not available in sample_generator_func")
             gen_src, gen_dst = gen_list

             # Loop indefinitely for tf.data
             while True:
                 try:
                     # Get ONE sample pair from each generator
                     # Assuming generate_next() can handle batch_size=1 request
                     # OR we need a get_one_sample() method
                     # Workaround: get a batch and yield one by one
                     src_samples, _ = gen_src.generate_next() # Get batch_size=1 (or more)
                     dst_samples, _ = gen_dst.generate_next()

                     # Extract individual samples from the batch
                     ws, ts, tsm_all, tsm_em = src_samples[0][0], src_samples[1][0], src_samples[2][0], src_samples[3][0]
                     wd, td, tdm_all, tdm_em = dst_samples[0][0], dst_samples[1][0], dst_samples[2][0], dst_samples[3][0]
                     # We can keep the explicit checks as a safeguard
                     if tsm_all.shape[-1] != 1: raise ValueError(f"tsm_all has wrong shape: {tsm_all.shape}")
                     if tsm_em.shape[-1] != 1: raise ValueError(f"tsm_em has wrong shape: {tsm_em.shape}")
                     if tdm_all.shape[-1] != 1: raise ValueError(f"tdm_all has wrong shape: {tdm_all.shape}")
                     if tdm_em.shape[-1] != 1: raise ValueError(f"tdm_em has wrong shape: {tdm_em.shape}")
                     yield (ws, ts, tsm_all, tsm_em, wd, td, tdm_all, tdm_em)

                 except StopIteration:
                      # Handle generator exhaustion if necessary (though they should loop)
                      io.log_info("Warning: A sample generator raised StopIteration.")
                      # Option 1: Re-initialize (might be complex)
                      # Option 2: Just break or yield dummy data? For now, break.
                      break # Or raise error if this shouldn't happen
                 except Exception as e:
                      io.log_err(f"Error in sample_generator_func: {e}")
                      traceback.print_exc()
                      # Decide how to handle errors during generation
                      # Maybe yield dummy data or raise? Re-raising for now.
                      raise e
        # --- End of Redefined sample_generator_func ---

        # !!! CRITICAL: Define the output_signature accurately !!!
        # Example based on shapes stored in model (verify these are correct B,H,W,C or B,C,H,W)
        res = model.resolution
        # Assuming floatx is set correctly in nn.py
        model_dtype = tf.keras.backend.floatx() # Get dtype from Keras backend
        bgr_shape_no_batch = (res, res, 3) if nn.data_format == "NHWC" else (3, res, res)
        mask_shape_no_batch = (res, res, 1) if nn.data_format == "NHWC" else (1, res, res)

        # --- Corrected output_signature for single samples ---
        bgr_shape_single = (res, res, 3) if nn.data_format == "NHWC" else (3, res, res)
        mask_shape_single = (res, res, 1) if nn.data_format == "NHWC" else (1, res, res)
        output_signature = (
             tf.TensorSpec(shape=bgr_shape_single, dtype=model_dtype), # warped_src
             tf.TensorSpec(shape=bgr_shape_single, dtype=model_dtype), # target_src
             tf.TensorSpec(shape=mask_shape_single, dtype=model_dtype),# target_srcm_all
             tf.TensorSpec(shape=mask_shape_single, dtype=model_dtype),# target_srcm_em
             tf.TensorSpec(shape=bgr_shape_single, dtype=model_dtype), # warped_dst
             tf.TensorSpec(shape=bgr_shape_single, dtype=model_dtype), # target_dst
             tf.TensorSpec(shape=mask_shape_single, dtype=model_dtype),# target_dstm_all
             tf.TensorSpec(shape=mask_shape_single, dtype=model_dtype),# target_dstm_em
        )
        # -------------------------------------------------------

        # --- Corrected Dataset Creation with batching ---
        dataset = tf.data.Dataset.from_generator(
             sample_generator_func, output_signature=output_signature
        ).batch(global_batch_size, drop_remainder=True) # <-- Re-add batching
        # ---------------------------------------------

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        dist_iterator = iter(dist_dataset)

        # --- Define the Distributed Training Step ---
        @tf.function
        def distributed_train_step(dist_inputs):
            # Removed apply_G_gradients flag logic
            per_replica_losses = strategy.run(model.train_step, args=(dist_inputs,)) # Removed kwargs
            # Aggregate losses
            agg_losses = {}
            for key in per_replica_losses.keys(): # Assuming train_step returns a dict
                 agg_losses[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[key], axis=None)
            return agg_losses

        # --- Training Loop ---
        is_reached_goal = model.is_reached_iter_goal()
        if model.get_target_iter() != 0:
            if is_reached_goal: io.log_info('Model already trained to target iteration.')
            else: io.log_info(f'Starting. Target iteration: {model.get_target_iter()}.')
        else: io.log_info('Starting.')
        if not no_preview: io.log_info('Press "Enter" to stop training and save model.')

        start_time = time.time()
        last_save_time = start_time
        last_preview_time = start_time
        save_interval_sec = save_interval_min * 60
        tensorboard_preview_interval_sec = 5 * 60

        # UI State
        selected_preview = 0
        zoom = Zoom.ZOOM_100
        show_preview = not no_preview
        wnd_name = "Training preview (TF2 Refactored)"
        previews_np = None # Hold last previews
        if show_preview: io.named_window(wnd_name); io.capture_keys(wnd_name);

        # Main loop using the model's optimizer iteration count
        while True:
            # --- Mark start of full iteration ---
            iter_start_wall_time = time.time()
            # ---------------------------------

            current_iter = model.get_iter()

            # --- Check for exit/goal ---
            if model.get_target_iter() != 0 and current_iter >= model.get_target_iter():
                 if not is_reached_goal:
                      io.log_info('\nReached target iteration.')
                      is_reached_goal = True
                      if not debug: io.log_info("Saving final model..."); checkpoint_manager.save(); io.log_info("Save completed.")
                 if no_preview: break # Exit if target reached and no preview

            # --- Callback: on_train_batch_begin ---
            # Removed LRDropoutCallback section
            # if model.is_training and not is_reached_goal and lr_dropout_callback is not None:
            #      lr_dropout_callback.on_train_batch_begin(current_iter) # Call manually

            # --- Get Data ---
            if current_iter == 10: # Profile one iteration early on
                pr = cProfile.Profile()
                pr.enable()
            try:
                 # Get data for all replicas
                 data_batch_dist = next(dist_iterator)
            except tf.errors.OutOfRangeError:
                 io.log_info("\nData generator exhausted. Reinitializing.")
                 dist_iterator = iter(dist_dataset) # Reinitialize iterator
                 continue
            except Exception as e:
                 io.log_err(f"\nError getting data batch: {e}")
                 traceback.print_exc(); break
            if current_iter == 10: # Stop profiling after the call finishes
                pr.disable()
                ps = pstats.Stats(pr).sort_stats('cumulative') # Sort by cumulative time
                ps.print_stats(30) # Print the top 30 time-consuming functions
                io.log_info("--- End CProfile Output ---")
                # Optionally exit after profiling:
                # break

            # --- Perform Training Step ---
            start_step_time = time.time() # Renamed from start_iter_time
            losses = {}
            if not is_reached_goal:
                try:
                     losses = distributed_train_step(data_batch_dist)
                     step_time = time.time() - start_step_time # Renamed from iter_time
                     # Add aggregated losses to history
                     model.add_loss_history([losses[key].numpy() for key in sorted(losses.keys()) if losses[key] is not None])
                except Exception as e:
                     io.log_err(f"\nError during training step: {e}")
                     traceback.print_exc()
                     # Optionally try to save before breaking
                     # if not debug: checkpoint_manager.save()
                     break # Stop training on error
            else:
                step_time = time.time() - start_step_time # Still measure time even if idle
                if show_preview: time.sleep(0.1)


            # --- Calculate Total Iteration Time ---
            # Place this calculation *after* all loop operations, including preview/sleep
            # (Need to calculate it *before* logging the string)
            total_iter_time = time.time() - iter_start_wall_time
            # ------------------------------------


            # --- Logging ---
            log_dict = {'iter': current_iter, 'step_time': step_time, 'total_iter_time': total_iter_time} # Log both times
            # Modify loss_string format
            loss_string = f"{time.strftime('[%H:%M:%S]')}[#{current_iter:06d}][S:{step_time*1000:04.0f}ms][T:{total_iter_time:05.2f}s]" # Added Total Time
            # Define keys to always exclude from the short log string
            exclude_keys_from_log = {'loss_G_gan', 'loss_G_true_face', 'loss_D_gan', 'loss_D_code'}
            for key, loss_tensor in losses.items():
                if loss_tensor is not None:
                    loss_val = loss_tensor.numpy()
                    log_dict[key] = loss_val # Log all to TensorBoard/history dict
                    # Only add to the short console string if not excluded
                    if key not in exclude_keys_from_log:
                         # Try to create a concise name
                         display_key = key.replace('loss_', '').replace('_mean', '').replace('G_', '').replace('D_', '')
                         # Limit length further if needed
                         display_key = display_key[:5] if len(display_key) > 5 else display_key
                         loss_string += f"[{display_key}:{loss_val:.4f}]"

            if not is_reached_goal:
                 if io.is_colab(): io.log_info('\r' + loss_string, end='')
                 else: io.log_info(loss_string, end='\r')

            # TensorBoard Logging (log both times)
            if not is_reached_goal and summary_writer is not None:
                with summary_writer.as_default(step=current_iter): # Set step context
                    tf.summary.scalar('time/step_gpu_ms', step_time * 1000) # Log step time in ms
                    tf.summary.scalar('time/total_iteration_s', total_iter_time) # Log total time in s
                    for key, loss_tensor in losses.items():
                         if loss_tensor is not None: tf.summary.scalar(f'loss/{key}', loss_tensor)
                    # Log learning rate
                    if hasattr(model, 'optimizer_G') and model.optimizer_G is not None:
                        lr_val = model.optimizer_G.learning_rate
                        if isinstance(lr_val, tf.keras.optimizers.schedules.LearningRateSchedule):
                            lr_val = lr_val(model.optimizer_G.iterations)
                        tf.summary.scalar('params/learning_rate', lr_val)

            # --- Periodic Saving ---
            cur_time = time.time()
            if not is_reached_goal and not debug and (cur_time - last_save_time) >= save_interval_sec:
                io.log_info(f"\nSaving model checkpoint at iteration {current_iter}...")
                checkpoint_manager.save()
                last_save_time = cur_time
                io.log_info("Save completed.")
                if not io.is_colab(): io.log_info(loss_string, end='\r')

            # --- Periodic Preview ---
            should_preview = False
            force_preview_now = False # Flag to force update if user presses 'p'
            if show_preview:
                 if (current_iter == 0 or is_reached_goal): should_preview = True
                 if (cur_time - last_preview_time) >= 30: should_preview = True # Preview every 30s

            if show_preview:
                key_events = io.get_key_events(wnd_name)
                key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0, 0, False, False, False)

                if key == ord('\n') or key == ord('\r'): io.log_info("\nStopping training and saving..."); break # Exit requested
                elif key == ord('s'): io.log_info("\nSaving model checkpoint manually..."); force_preview_now=True;
                elif key == ord('b'): io.log_info("\nCreating backup (saving checkpoint)..."); force_preview_now=True;
                elif key == ord('p'): force_preview_now = True;
                elif key == ord(' '):
                     if previews_np: selected_preview = (selected_preview + 1) % len(previews_np)
                     force_preview_now = True # Force redraw with new selection
                elif chr_key == '-': zoom = zoom.prev(); force_preview_now = True;
                elif chr_key == '+' or chr_key == '=': zoom = zoom.next(); force_preview_now = True;

                if key == ord('s') or key == ord('b'):
                    if not debug: checkpoint_manager.save(); io.log_info("Save completed.")

            # TensorBoard image logging interval
            if summary_writer is not None and not is_reached_goal and (cur_time - last_preview_time) >= tensorboard_preview_interval_sec:
                should_preview = True # Also trigger for TB

            if should_preview or force_preview_now:
                 last_preview_time = cur_time
                 try:
                     preview_data = model.get_preview_samples()
                     previews_np = model.onGetPreview(preview_data)

                     if previews_np:
                         # UI Preview
                         if show_preview:
                              preview_pane_image = create_preview_pane_image(previews_np, selected_preview,
                                                                             model.get_loss_history(), 0, # History range disabled for now
                                                                             current_iter, model.batch_size, zoom)
                              io.show_image(wnd_name, preview_pane_image)

                         # TensorBoard Preview
                         if summary_writer is not None:
                             with summary_writer.as_default(step=current_iter): # Set step context
                                 for i, (name, img_np) in enumerate(previews_np):
                                     img_tensor = tf.expand_dims(img_np, 0) # Add batch dim
                                     tf.summary.image(f"preview/{name}", img_tensor, max_outputs=1)
                 except Exception as e:
                      io.log_err(f"\nError generating preview: {e}")
                      traceback.print_exc()


            # --- Process GUI messages and sleep ---
            if show_preview:
                try: io.process_messages(0.01) # Short sleep for UI responsiveness
                except KeyboardInterrupt: io.log_info("\nKeyboardInterrupt received, stopping."); break
            elif current_iter % 100 == 0: # Still process messages occasionally without UI
                 try: io.process_messages(0.001)
                 except KeyboardInterrupt: io.log_info("\nKeyboardInterrupt received, stopping."); break

            # Recalculate total_iter_time *just before potentially printing the final log line*
            # This is needed if the preview/sleep significantly affects the time shown on the next line break
            total_iter_time = time.time() - iter_start_wall_time
            # (The loss_string uses the total_iter_time calculated earlier in the loop for the end='\r' print)

        # --- End of Training Loop ---
        if show_preview: io.destroy_all_windows()
        if summary_writer is not None: summary_writer.close()
        # model.finalize() # Call if model has specific cleanup
        # Final save if user interrupted training
        if not is_reached_goal and not debug: # Only save if goal wasn't reached (i.e. user interrupted) and not in debug
            io.log_info("\nSaving final model checkpoint...")
            try:
                checkpoint_manager.save()
                io.log_info("Final save completed.")
            except Exception as e:
                io.log_err(f"Error during final save: {e}")
        elif is_reached_goal:
            io.log_info("\nTarget iteration reached. Final model already saved during periodic saving or before loop exit.")

        io.log_info("\nTrainer finished.")

    except Exception as e:
        io.log_err(f'\nError: {str(e)}')
        traceback.print_exc()


# --- END OF FILE mainscripts/Trainer.py ---