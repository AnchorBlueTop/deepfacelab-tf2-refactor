# --- START OF FILE core/leras/optimizers/RMSprop_refactored.py ---

import tensorflow as tf
import numpy as np

# Inherit from the standard Keras Optimizer base class
class RMSprop(tf.keras.optimizers.Optimizer):
    """
    Custom RMSprop Optimizer implementing original leras features:
    - lr_dropout: Randomly masks gradient updates based on a probability.
    - Cosine LR Decay integration (via Keras schedules).
    - TF1-style rho parameter.
    """
    def __init__(self,
                 learning_rate=0.001,
                 rho=0.9, # Decay rate for the moving average of squared gradients
                 lr_dropout_rate=1.0, # Probability of *keeping* an update (1.0 = disabled)
                 epsilon=1e-7, # Small constant for stability (Keras default)
                 clipnorm=None, # Use Keras standard clipnorm
                 clipvalue=None,# Use Keras standard clipvalue
                 name="LerasRMSprop", # Changed default name
                 **kwargs):
        """
        Args:
            learning_rate: Float or tf.keras.optimizers.schedules.LearningRateSchedule.
            rho: Float. Discounting factor for the history/coming gradient.
            lr_dropout_rate: Float between 0.0 and 1.0. Probability of applying
                             the calculated gradient update to a weight. If 1.0,
                             dropout is disabled.
            epsilon: Float. Small scalar value to avoid division by zero.
            clipnorm: Float. If set, clips gradients by L2 norm.
            clipvalue: Float. If set, clips gradients by value.
            name: String. The name of the optimizer instance.
            **kwargs: Keyword arguments. Allowed to be {`decay`}. `decay` is ignored.
        """
        # Initialize the parent Keras Optimizer
        super().__init__(name=name, clipnorm=clipnorm, clipvalue=clipvalue, **kwargs)

        # Store hyperparameters specific to this implementation
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate)) # Allow 'lr' for compatibility
        self._set_hyper('rho', rho)
        self._set_hyper('lr_dropout_rate', lr_dropout_rate) # Store dropout probability

        # Epsilon handling (use Keras internal if possible, otherwise store)
        self._epsilon = epsilon or tf.keras.backend.epsilon()

        # Cosine decay is now handled by passing a Keras schedule as learning_rate
        if 'lr_cos' in kwargs and kwargs['lr_cos'] > 0:
            print("Warning: 'lr_cos' is deprecated. Pass a tf.keras.optimizers.schedules.CosineDecay "
                  "object as 'learning_rate' instead.")

        # LR dropout on CPU is no longer relevant/controllable this way


    # Create optimizer slots (moving average of squared gradients)
    def _create_slots(self, var_list):
        for var in var_list:
            # 'rms' is the conventional name for the moving average slot in Keras RMSprop
            self.add_slot(var, 'rms')

    # Implement the core update logic for dense variables
    @tf.function # Decorate with tf.function for performance
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # Get current LR (handles schedules)
        rms_var = self.get_slot(var, 'rms') # Get the moving average slot
        rho_t = self._get_hyper('rho', var_dtype)
        lr_dropout_rate_t = self._get_hyper('lr_dropout_rate', var_dtype)
        epsilon_t = tf.cast(self._epsilon, var_dtype)

        # --- RMSprop Update ---
        # rms_t = rms * rho + (1 - rho) * grad^2
        new_rms = rho_t * rms_var + (1.0 - rho_t) * tf.square(grad)

        # var_t = var - lr * grad / (sqrt(rms_t) + epsilon)
        v_diff = lr_t * grad / (tf.sqrt(new_rms) + epsilon_t)

        # --- Apply Learning Rate Dropout ---
        if lr_dropout_rate_t < 1.0:
            # Generate random binomial mask *at each step*
            keep_mask = tf.keras.backend.random_binomial(
                shape=tf.shape(grad), p=lr_dropout_rate_t, dtype=var_dtype
            )
            v_diff = v_diff * keep_mask # Apply mask

        # --- Update Variable and Slot ---
        # Update the moving average slot first
        rms_update = rms_var.assign(new_rms, use_locking=self._use_locking)
        
        # Then update the variable: var = var - v_diff (or var = var + (-v_diff))
        var_update = var.assign_sub(v_diff, use_locking=self._use_locking)

        # Standard Keras practice: return the main variable update op for dependency tracking
        return var_update

    # Implement sparse update (optional but good practice)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        rms_var = self.get_slot(var, 'rms')
        rho_t = self._get_hyper('rho', var_dtype)
        lr_dropout_rate_t = self._get_hyper('lr_dropout_rate', var_dtype)
        epsilon_t = tf.cast(self._epsilon, var_dtype)

        # --- RMSprop Update for Sparse Gradients ---
        rms_slice = tf.gather(rms_var, indices)
        new_rms_slice = rho_t * rms_slice + (1.0 - rho_t) * tf.square(grad)

        v_diff_slice = lr_t * grad / (tf.sqrt(new_rms_slice) + epsilon_t)

        # --- Apply Learning Rate Dropout ---
        if lr_dropout_rate_t < 1.0:
            keep_mask_slice = tf.keras.backend.random_binomial(
                shape=tf.shape(grad), p=lr_dropout_rate_t, dtype=var_dtype
            )
            v_diff_slice = v_diff_slice * keep_mask_slice

        # --- Update Variable and Slot ---
        # Update the rms slot first
        rms_update = state_ops.scatter_update(rms_var, indices, new_rms_slice, use_locking=self._use_locking)
        
        # Then update the variable using scatter_sub
        var_update = state_ops.scatter_sub(var, indices, v_diff_slice, use_locking=self._use_locking)

        # Standard Keras practice: return the main variable update op
        return var_update


    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'rho': self._serialize_hyperparameter('rho'),
            'lr_dropout_rate': self._serialize_hyperparameter('lr_dropout_rate'),
            'epsilon': self._epsilon,
            # clipnorm and clipvalue are handled by base class get_config
        })
        return config

# Assign the refactored class back to nn.RMSprop for compatibility
# nn.RMSprop = RMSprop
# --- END OF FILE core/leras/optimizers/RMSprop_refactored.py ---