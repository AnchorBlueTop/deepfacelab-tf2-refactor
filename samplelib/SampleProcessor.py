import collections
import math
from enum import IntEnum
from core.imagelib.shadows import shadow_highlights_augmentation

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import sd, LinearMotionBlur
from core.imagelib.color_transfer import random_lab_rotation
from facelib import FaceType, LandmarksProcessor


class SampleProcessor(object):
    class SampleType(IntEnum):
        NONE = 0
        IMAGE = 1
        FACE_IMAGE = 2
        FACE_MASK  = 3
        LANDMARKS_ARRAY            = 4
        PITCH_YAW_ROLL             = 5
        PITCH_YAW_ROLL_SIGMOID     = 6

    class ChannelType(IntEnum):
        NONE = 0
        BGR                   = 1  #BGR
        G                     = 2  #Grayscale
        GGG                   = 3  #3xGrayscale
        LAB_RAND_TRANSFORM    = 4  # LAB random transform


    class FaceMaskType(IntEnum):
        NONE           = 0
        FULL_FACE      = 1  # mask all hull as grayscale
        EYES           = 2  # mask eyes hull as grayscale
        FULL_FACE_EYES = 3  # eyes and mouse

    class Options(object):
        class AugmentationParams(object): # Keep nested definition
            def __init__(self, random_flip=False, ct_mode=None, random_ct_mode=False,
                        random_hsv_power=0.0, random_downsample=False, random_noise=False,
                        random_blur=False, random_jpeg=False, random_shadow='none'):
                self.random_flip = random_flip # Note: This seems redundant with Options.random_flip
                self.ct_mode = ct_mode
                self.random_ct_mode = random_ct_mode
                self.random_hsv_power = random_hsv_power
                self.random_downsample = random_downsample
                self.random_noise = random_noise
                self.random_blur = random_blur
                self.random_jpeg = random_jpeg
                self.random_shadow = random_shadow

        # --- Updated __init__ for Options ---
        def __init__(self, batch_size=1, # Keep batch_size? Seems unused by process() itself
                        resolution=0,
                        face_type=None, # Expects FaceType Enum
                        mask_type=None, # Expects FaceMaskType Enum
                        eye_mouth_prio=False,
                        augmentation_params=None, # Expects AugmentationParams object
                        random_warp=False,
                        true_face_power=0.0,
                        # Keep original geometric args too
                        random_flip=True,
                        rotation_range=[-2,2],
                        scale_range=[-0.05, 0.05],
                        tx_range=[-0.05, 0.05],
                        ty_range=[-0.05, 0.05] ):

            # Store all passed arguments as attributes
            self.batch_size = batch_size
            self.resolution = resolution
            self.face_type = face_type
            self.mask_type = mask_type
            self.eye_mouth_prio = eye_mouth_prio
            # Create default AugmentationParams if None is passed, using nested class access
            self.augmentation_params = augmentation_params if augmentation_params is not None \
                                       else self.AugmentationParams() # Use self.AugmentationParams
            self.random_warp = random_warp
            self.true_face_power = true_face_power
            # Store original geometric args
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range

    @staticmethod
    def process (samples, sample_process_options, output_sample_types, debug, ct_sample=None, rnd_state=None, face_scale=1.0): # Added rnd_state, face_scale based on SampleGeneratorFace usage
        SPST = SampleProcessor.SampleType
        SPCT = SampleProcessor.ChannelType
        SPFMT = SampleProcessor.FaceMaskType

        outputs = []
        for sample in samples:
            # Initialize random state for this sample if not provided
            if rnd_state is None:
                 sample_rnd_seed = np.random.randint(0x80000000)
                 rnd_state = np.random.RandomState(sample_rnd_seed)
            else: # Use passed state and derive sample seed maybe?
                sample_rnd_seed = rnd_state.randint(0x80000000)

            # --- Load Base Sample Data ---
            sample_face_type = sample.face_type # This is the sample's original FaceType Enum
            sample_bgr = sample.load_bgr()
            sample_landmarks = sample.landmarks
            ct_sample_bgr = None # Initialize ct_sample_bgr
            h,w,c = sample_bgr.shape
            is_face_sample = sample_landmarks is not None

            # --- Define Mask Helper Functions (closures capture sample variables) ---
            def get_full_face_mask():
                xseg_mask = sample.get_xseg_mask()
                if xseg_mask is not None:
                    if xseg_mask.shape[0] != h or xseg_mask.shape[1] != w:
                        xseg_mask = cv2.resize(xseg_mask, (w,h), interpolation=cv2.INTER_CUBIC)
                        xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
                    return np.clip(xseg_mask, 0, 1)
                else:
                    full_face_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                    return np.clip(full_face_mask, 0, 1)

            def get_eyes_mask():
                eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                clip = np.clip(eyes_mask, 0, 1)
                clip[clip > 0.1] += 1 # Value 2 for eyes
                return clip

            def get_mouth_mask():
                mouth_mask = LandmarksProcessor.get_image_mouth_mask (sample_bgr.shape, sample_landmarks)
                clip = np.clip(mouth_mask, 0, 1)
                clip[clip > 0.1] += 2 # Value 3 for mouth (assuming eyes used 2)
                return clip
            # ---------------------------------------------------------------------

            if debug and is_face_sample:
                LandmarksProcessor.draw_landmarks (sample_bgr, sample_landmarks, (0, 1, 0))

            outputs_sample = []
            for opts in output_sample_types:
                # --- Get Options for this output type ---
                resolution     = opts.get('resolution', 0)
                sample_type    = opts.get('sample_type', SPST.NONE)
                channel_type   = opts.get('channel_type', SPCT.NONE)
                face_type_str  = opts.get('face_type', None) # Get the target face type STRING
                face_mask_type = opts.get('face_mask_type', SPFMT.NONE) # Get target mask type
                nearest_resize_to = opts.get('nearest_resize_to', None)
                warp           = opts.get('warp', False)
                transform      = opts.get('transform', False)
                normalize_tanh = opts.get('normalize_tanh', False)
                ct_mode        = opts.get('ct_mode', None)
                data_format    = opts.get('data_format', 'NHWC')
                border_replicate = opts.get('border_replicate', sample_type == SPST.FACE_IMAGE) # Default border replicate for face images

                # Augmentation options
                aug_params = sample_process_options.augmentation_params # Get from overall options
                random_hsv_shift_amount = aug_params.random_hsv_power if aug_params else 0.0 # Example access
                random_downsample = aug_params.random_downsample if aug_params else False
                random_noise = aug_params.random_noise if aug_params else False
                random_blur = aug_params.random_blur if aug_params else False
                random_jpeg = aug_params.random_jpeg if aug_params else False
                random_shadow_type = aug_params.random_shadow if aug_params else 'none'

                # Random state seeds
                rnd_seed_shift      = opts.get('rnd_seed_shift', 0)
                warp_rnd_seed_shift = opts.get('warp_rnd_seed_shift', rnd_seed_shift)
                sample_specific_rnd_state = np.random.RandomState (sample_rnd_seed+rnd_seed_shift)
                warp_rnd_state            = np.random.RandomState (sample_rnd_seed+warp_rnd_seed_shift)
                # -----------------------------------------

                # --- Convert Face Type String to Enum ---
                target_face_type_enum = None
                if face_type_str is not None:
                    try:
                        # Ensure FaceType is imported at the top of this file
                        # from facelib import FaceType
                        target_face_type_enum = FaceType.fromString(face_type_str)
                    except Exception as e:
                         print(f"Warning: Invalid face_type string '{face_type_str}' in output_sample_types. Error: {e}")
                         # If conversion fails, target_face_type_enum remains None
                # ---------------------------------------

                # Determine warp params based on the specific warp random state
                warp_params = imagelib.gen_warp_params(resolution,
                                                       sample_process_options.random_flip,
                                                       rotation_range=sample_process_options.rotation_range,
                                                       scale_range=sample_process_options.scale_range,
                                                       tx_range=sample_process_options.tx_range,
                                                       ty_range=sample_process_options.ty_range,
                                                       rnd_state=rnd_state, # Use main random state for affine?
                                                       warp_rnd_state=warp_rnd_state # Use specific state for grid warp
                                                       )

                # Determine border mode
                borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT

                # --- Process based on Sample Type ---
                if sample_type == SPST.FACE_MASK or sample_type == SPST.FACE_IMAGE:
                    if not is_face_sample:
                        raise ValueError("face_samples should be provided for sample_type FACE_*")
                    if target_face_type_enum is None:
                         raise ValueError(f"Valid face_type enum could not be determined for string '{face_type_str}'")

                    if sample_type == SPST.FACE_MASK:
                        # --- Generate the appropriate base mask ---
                        if face_mask_type == SPFMT.FULL_FACE:
                            img = get_full_face_mask()
                        elif face_mask_type == SPFMT.EYES: # Not typically used directly? Check output_sample_types
                            img = get_eyes_mask()
                            # Original logic added 1 after clip, then checked > 1. Simpler is just the mask.
                            # img = np.clip(img, 0, 1) # Just return the eye mask
                        elif face_mask_type == SPFMT.FULL_FACE_EYES:
                            # Get base face mask
                            img = get_full_face_mask()
                            mask = img.copy()
                            mask[mask != 0.0] = 1.0; # Create binary mask
                            # Get eye/mouth masks and combine using original value scheme (1=face, 2=eyes, 3=mouth)
                            eye_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                            eye_mask = np.clip(eye_mask, 0, 1)
                            eye_mask[eye_mask > 0.1] = 2 # Value 2 for eyes
                            img = np.where(eye_mask > 1, eye_mask, img) # Add eyes

                            mouth_mask = LandmarksProcessor.get_image_mouth_mask (sample_bgr.shape, sample_landmarks)
                            mouth_mask = np.clip(mouth_mask, 0, 1)
                            mouth_mask[mouth_mask > 0.1] = 3 # Value 3 for mouth
                            img = np.where(mouth_mask > 2, mouth_mask, img) # Add mouth
                            img *= mask # Re-apply binary face mask to remove artifacts outside hull

                        else: # SPFMT.NONE or other
                            img = np.zeros ( sample_bgr.shape[0:2]+(1,), dtype=np.float32)

                        # --- Apply geometric transform for alignment ---
                        # Check if alignment is needed based on target vs sample type
                        if target_face_type_enum != sample_face_type:
                            mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, target_face_type_enum)
                            img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_LINEAR )
                        else: # If face types match, just resize if needed (should have shape H,W,1 or H,W)
                            if img.shape[0] != resolution or img.shape[1] != resolution:
                                img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LINEAR )
                                if img.ndim == 2: img = img[...,None] # Add channel dim if lost by resize

                        # --- Apply warp/transform/flip using warp_params ---
                        img = imagelib.warp_by_params (warp_params, img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)

                        # --- Final processing for mask ---
                        # Ensure correct shape and channel count (should be H, W, 1)
                        if img.ndim == 2: img = img[...,None]
                        if img.shape[-1] != 1:
                            # This should ideally not happen after LandmarksProcessor fixes, but as safety:
                            print(f"Warning: Correcting mask channels from {img.shape[-1]} to 1 in SampleProcessor.")
                            img = img[..., 0:1] # Take first channel

                        # Ensure correct output channel type
                        if channel_type == SPCT.G:
                            out_sample = img.astype(np.float32)
                        else:
                            # This case shouldn't be hit based on generator options
                            print(f"Warning: Unexpected channel_type {channel_type} requested for FACE_MASK. Returning grayscale.")
                            out_sample = img.astype(np.float32) # Default to returning grayscale

                    elif sample_type == SPST.FACE_IMAGE:
                        img = sample_bgr

                        # Apply geometric transform to align face image
                        if target_face_type_enum != sample_face_type:
                            mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, target_face_type_enum) # Use Enum
                            img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_CUBIC )
                        else: # If face types match, just resize if needed
                            if w != resolution: img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_CUBIC )

                        # Apply random color transfer if enabled and ct_sample provided
                        if ct_mode is not None and ct_sample is not None:
                            if ct_sample_bgr is None: ct_sample_bgr = ct_sample.load_bgr() # Load only once per sample pair
                            if ct_sample_bgr is not None:
                                img = imagelib.color_transfer (ct_mode, img, cv2.resize( ct_sample_bgr, (resolution,resolution), interpolation=cv2.INTER_LINEAR ) )
                        elif ct_mode == 'fs-aug': # FS-Aug mode uses augmentations
                             img = imagelib.color_augmentation(img, sample_rnd_seed)

                        # --- Apply Augmentations ---
                        # Apply random warp/transform/flip first
                        img  = imagelib.warp_by_params (warp_params, img,  warp, transform, can_flip=True, border_replicate=border_replicate)

                        # Apply other distortions in random order
                        randomization_order = []
                        if random_blur: randomization_order.append('blur')
                        if random_noise: randomization_order.append('noise')
                        if random_jpeg: randomization_order.append('jpeg')
                        if random_downsample: randomization_order.append('down')
                        sample_specific_rnd_state.shuffle(randomization_order)

                        for random_distortion in randomization_order:
                            if random_distortion == 'blur':
                                blur_type = sample_specific_rnd_state.choice(['motion', 'gaussian'])
                                if blur_type == 'motion': blur_k = sample_specific_rnd_state.randint(5, 12); blur_angle = 360 * sample_specific_rnd_state.rand(); img = LinearMotionBlur(img, blur_k, blur_angle)
                                elif blur_type == 'gaussian': blur_sigma = 3 * sample_specific_rnd_state.rand() + 1; ks = int(2.9 * blur_sigma); ks += 1 if ks % 2 == 0 else 0; img = cv2.GaussianBlur(img, (ks,ks), blur_sigma)
                            elif random_distortion == 'noise':
                                noise_type = sample_specific_rnd_state.choice(['gaussian', 'laplace', 'poisson'])
                                noise_scale = (10 * sample_specific_rnd_state.rand() + 10) / 255.0
                                if noise_type == 'gaussian': noise = sample_specific_rnd_state.normal(scale=noise_scale, size=img.shape)
                                elif noise_type == 'laplace': noise = sample_specific_rnd_state.laplace(scale=noise_scale, size=img.shape)
                                elif noise_type == 'poisson': noise_lam = (10 * sample_specific_rnd_state.rand() + 10); noise = sample_specific_rnd_state.poisson(lam=noise_lam, size=img.shape) / 255.0 - (noise_lam/255.0)
                                img = np.clip(img + noise, 0, 1)
                            elif random_distortion == 'jpeg':
                                q = sample_specific_rnd_state.randint(60, 95); ret, buf = cv2.imencode('.jpg', (img*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), q]); img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
                            elif random_distortion == 'down':
                                interp = sample_specific_rnd_state.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA])
                                down_res = sample_specific_rnd_state.randint(int(0.25*resolution), int(0.75*resolution)); img = cv2.resize(img, (down_res, down_res), interpolation=interp); img = cv2.resize(img, (resolution, resolution), interpolation=interp);

                        # Random HSV shift
                        if random_hsv_shift_amount != 0:
                            a = random_hsv_shift_amount; h_amount = max(1, int(180*a*0.5));
                            img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                            img_h = (img_h + sample_specific_rnd_state.randint(-h_amount, h_amount+1) ) % 180 # Hue wraps at 180 in OpenCV
                            img_s = np.clip (img_s + (sample_specific_rnd_state.random()-0.5)*a, 0, 1 )
                            img_v = np.clip (img_v + (sample_specific_rnd_state.random()-0.5)*a, 0, 1 )
                            img = np.clip( cv2.cvtColor(cv2.merge([img_h, img_s, img_v]), cv2.COLOR_HSV2BGR) , 0, 1 )

                        # Random shadow
                        if random_shadow_type != 'none' and sample_specific_rnd_state.randint(2) == 0:
                            img = shadow_highlights_augmentation(img, sample_rnd_seed)

                        # Final clip after all augmentations
                        img = np.clip(img, 0, 1)

                        # Transform from BGR to desired channel_type
                        if channel_type == SPCT.BGR: out_sample = img
                        elif channel_type == SPCT.LAB_RAND_TRANSFORM: out_sample = random_lab_rotation(img, sample_rnd_seed)
                        elif channel_type == SPCT.G: out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                        elif channel_type == SPCT.GGG: out_sample = np.repeat ( np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1), (3,), -1)
                        else: out_sample = img # Default to BGR if channel type unknown

                elif sample_type == SPST.IMAGE:
                    img = sample_bgr
                    img = imagelib.warp_by_params (warp_params, img, warp, transform, can_flip=True, border_replicate=True)
                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                    out_sample = img

                elif sample_type == SPST.LANDMARKS_ARRAY:
                    l = sample_landmarks
                    # Warp landmarks
                    l = LandmarksProcessor.transform_points(l, warp_params['rmat'])
                    if warp_params['flip']: l = LandmarksProcessor.mirror_landmarks(l, w)
                    # Normalize
                    l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                    l = np.clip(l, 0.0, 1.0)
                    out_sample = l.astype(np.float32)

                elif sample_type == SPST.PITCH_YAW_ROLL or sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                    pitch, yaw, roll = sample.get_pitch_yaw_roll()
                    if warp_params['flip']: yaw = -yaw

                    if sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                         pitch = np.clip( (pitch / math.pi) / 2.0 + 0.5, 0, 1)
                         yaw   = np.clip( (yaw / math.pi) / 2.0 + 0.5, 0, 1)
                         roll  = np.clip( (roll / math.pi) / 2.0 + 0.5, 0, 1)

                    out_sample = np.array([pitch, yaw, roll], dtype=np.float32)
                else:
                    raise ValueError ('expected sample_type')

                # Final transformations common to most types
                if nearest_resize_to is not None:
                     out_sample = cv2_resize(out_sample, (nearest_resize_to,nearest_resize_to), interpolation=cv2.INTER_NEAREST)

                if normalize_tanh: # Apply tanh normalization [-1, 1]
                     out_sample = np.clip (out_sample * 2.0 - 1.0, -1.0, 1.0)

                # Final data format transpose if needed
                if data_format == "NCHW" and out_sample.ndim == 3: # Only transpose images
                     out_sample = np.transpose(out_sample, (2,0,1) )

                outputs_sample.append(out_sample)

            outputs += [outputs_sample] # Append list of outputs for this sample

        # Return list of lists of processed samples
        return outputs