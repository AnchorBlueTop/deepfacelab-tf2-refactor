# DeepFaceLab TensorFlow 2.x Migration: A Study in Framework Refactoring and VRAM Optimization

A comprehensive refactoring of the DeepFaceLab training pipeline from TensorFlow 1.x compatibility mode to native TensorFlow 2.x architecture. This project successfully achieved significant VRAM reduction enabling substantially larger model configurations, though replicating the original implementation's nuanced training dynamics presented unexpected challenges.

## Project Status

**Achieved:** 60% VRAM reduction, enabling 3x resolution increase (416x416 → 704x704 at batch size 16)

**Challenge:** Training quality did not achieve parity with original TF1 implementation despite numerical stability

This repository documents both the technical achievements and the obstacles encountered during migration, providing insights into the complexities of modernizing deep learning frameworks.

## Background

DeepFaceLab is a deep learning framework for facial reenactment and face swapping, primarily using autoencoder architectures. The MVE (Mod VAE Extended) fork extended the original framework with additional features but remained built on TensorFlow 1.x compatibility APIs running under TensorFlow 2.6.0.

### Initial Problem Statement

Training on an NVIDIA RTX 5090 (32GB VRAM) consistently hit Out of Memory errors when attempting to train at desired configurations:
- Resolution: 544px or higher
- Batch size: 12+ 
- Model dimensions sufficient for high-quality output

The existing TF1-style gradient checkpointing implementation (`memory_saving_gradients.py`) provided no noticeable VRAM savings. The primary motivation for this refactoring was to leverage TensorFlow 2.x's native `tf.recompute_grad` for gradient checkpointing while modernizing the codebase to current TensorFlow practices.

## Understanding the Fundamentals

Before diving into the technical details, it's helpful to understand the core concepts behind DeepFaceLab and deep learning systems in general.

### What is a Neural Network?

A **neural network** is essentially a sophisticated pattern-matching system modeled loosely on how brains work. Instead of following explicit rules (like traditional programming), neural networks *learn* patterns from examples.

Think of it like this: if you wanted to teach a computer to recognize cats, you wouldn't write rules like "if it has pointy ears and whiskers, it's a cat." Instead, you'd show the network thousands of cat pictures, and it would gradually learn the patterns that distinguish cats from dogs, cars, or anything else.

A neural network consists of:
- **Layers** of artificial "neurons" (just mathematical functions)
- **Weights** (numbers that get adjusted during learning)
- **Connections** between layers that transform data

### How DeepFaceLab Works: Face Swapping Explained

**The Core Idea:**
DeepFaceLab uses a special type of neural network called an **autoencoder** to learn how to represent and reconstruct faces. Here's the process:

1. **Compression (Encoder):** Take a face image and compress it down to a small set of numbers (called a "latent code") that capture the essence of that face - things like pose, expression, lighting, and identity.

2. **Processing (Inter):** The latent code can be manipulated or processed in this compressed form.

3. **Reconstruction (Decoder):** Take that small set of numbers and reconstruct them back into a full face image.

The "magic" of face swapping happens because:
- You train **two decoders**: one that learned to reconstruct Person A's face, and one that learned Person B's face
- When you want to swap faces, you take Person A's face, compress it to a latent code, then feed that code into Person B's decoder
- Person B's decoder reconstructs a face with Person A's expressions/pose but Person B's identity features

**Visual Analogy:**
Imagine the encoder as someone describing a face to you over the phone (compressed information), and the decoder as you drawing the face based on that description. If you trained a decoder to only know how to draw Person B's face, and someone describes Person A's expression to you, you'd draw Person B making Person A's expression.

### What Are Gradients and Why Do They Matter?

When training a neural network, you need to adjust the weights to make better predictions. But with millions of weights, how do you know which ones to adjust and by how much?

This is where **gradients** come in:
- A **gradient** is like an arrow that points in the direction of "improvement"
- It tells you: "if you change this weight by a small amount in this direction, the network's output will get better"
- The **gradient's magnitude** tells you how sensitive the output is to changes in that weight

**Training Process:**
1. The network makes a prediction (forward pass)
2. You calculate how wrong it was (loss)
3. You compute gradients: "how should each weight change to reduce this error?"
4. You update all the weights by small steps in the direction of their gradients (backpropagation)
5. Repeat millions of times until the network gets good

**Why This Project Cared About Gradients:**
The refactoring project spent significant time debugging gradient flow because:
- If gradients are too small ("vanishing gradients"), some layers stop learning
- If gradients are too large ("exploding gradients"), training becomes unstable
- The weight scaling technique (wscale) directly affects gradient magnitudes
- Different frameworks (TF1 vs TF2) compute gradients in subtly different ways

### What is VRAM and Why Did It Matter?

**VRAM (Video RAM)** is the memory on your graphics card (GPU). Deep learning training needs massive amounts of memory to store:
- All the layer activations (intermediate results) during the forward pass
- All the gradients during the backward pass
- The model weights themselves
- Batch of training images

Higher resolution images and larger batch sizes require more VRAM. The RTX 5090 has 32GB VRAM, which sounds like a lot, but:
- A 704x704 image has 3x more pixels than a 412x412 image
- Training an autoencoder stores activations for *every layer* of both encoder and decoder
- Running out of VRAM (Out Of Memory errors) means you can't train at your desired settings

The entire motivation for this refactoring was to reduce VRAM usage so larger, higher-quality models could be trained.

### What is TensorFlow and What Changed Between Versions?

**TensorFlow** is Google's deep learning framework - essentially a library of tools for building and training neural networks.

**TensorFlow 1.x Approach (what OG DFL used):**
- Build a static "computation graph" upfront (like drawing a flowchart)
- Run that graph in a "session" by feeding data through it
- More explicit control but more complex code

**TensorFlow 2.x Approach (what the refactor targeted):**
- "Eager execution" - operations run immediately, like normal Python
- Keras API provides simpler, more intuitive layer/model classes
- Graph compilation happens automatically when needed
- More user-friendly but less explicit control

The refactoring challenge was that **OG DFL's training behavior was tightly coupled to TF1's specific way of building graphs**. Even though the math was identical, TF2's different execution model produced different training dynamics.

### Understanding DeepFaceLab's Architecture

Before diving into the refactoring challenges, it's important to understand the core architecture:

**Autoencoder-Based Face Swapping**
- **Encoder:** Compresses input face images into lower-dimensional latent representations (latent codes)
- **Inter (Intermediate):** Processes and manipulates the latent space
- **Decoder:** Reconstructs face images from latent codes

**Two Training Modes:**
- `df` mode: Separate decoders for source and destination faces
- `liae` mode: Shared decoder with separate intermediate blocks for identity disentanglement

**Face Swapping Mechanism:** Feed source face's latent code into destination's decoder (or vice versa), leveraging the learned latent space to generate realistic face transfers.

This architecture's success heavily depends on stable training dynamics and proper variance propagation through the network - challenges that became central to the refactoring effort.

## Understanding Equalized Learning Rate (Weight Scaling)

One of the most critical components of the original DFL implementation was its use of **equalized learning rate** (also known as weight scaling or "wscale"), a technique popularized by Progressive GAN research. Understanding this technique is essential to appreciating both the refactoring challenges and the learning outcomes from this project.

### The Theory

Equalized learning rate addresses a fundamental problem in deep neural network training: different layers can have vastly different effective learning rates due to variations in their parameter initialization scales.

**Standard Initialization Problem:**
- Typical initialization schemes (He, Xavier) scale initial weights based on fan-in: `W ~ N(0, std)` where `std = gain / sqrt(fan_in)`
- Layers with different fan-in values start with different weight magnitudes
- During training, gradient updates have different relative impacts on different layers
- Some layers may dominate learning while others lag behind

**Weight Scaling Solution:**
1. Initialize ALL weights from standard normal distribution: `W ~ N(0, 1.0)`
2. Calculate runtime scale factor: `scale = gain / sqrt(fan_in)`
3. Apply scaling during forward pass: `output = conv(input, W * scale)`
4. Never modify the stored weights by the scale factor

**Key Benefits:**
- All layers have the same magnitude gradients relative to their weights
- More balanced learning dynamics across the network
- Better stability during training, especially for deeper networks
- Original DFL relied heavily on this for its fast initial convergence

### Original DFL TF1 Implementation

In the original DeepFaceLab TF1 codebase:

```python
class Conv2D(LayerBase):  # TF1 style
    def build_weights(self):
        # Initialize from N(0,1) if use_wscale is True
        self.weight = tf.get_variable(..., initializer=tf.random_normal(0, 1.0))
        
        if self.use_wscale:
            fan_in = kernel_size * kernel_size * in_channels
            he_std = gain / np.sqrt(fan_in)
            self.wscale = tf.constant(he_std)  # Store as constant
    
    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale  # Runtime scaling
        return tf.nn.conv2d(x, weight, ...)
```

Crucially, in TF1's graph construction:
- The multiplication `weight * self.wscale` creates a distinct graph operation
- The scaled weight tensor flows into `tf.nn.conv2d`
- TF1's static graph construction and specific graph op linking produced stable behavior
- The decoder output typically had `tf.nn.sigmoid` applied **within the Decoder's forward method**, naturally producing [0,1] range outputs

### TF2 Keras Subclassing Challenges

The refactored TF2 implementation attempted to replicate this:

```python
class WScaleConv2D(tf.keras.layers.Conv2D):  # TF2 Keras style
    def __init__(self, filters, kernel_size, gain=math.sqrt(2.0), **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer=Zeros(),
            **kwargs
        )
        self.gain = gain
        self.runtime_scale = None

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.prod(self.kernel.shape[:-1])
        self.runtime_scale = self.gain * tf.math.rsqrt(tf.cast(fan_in, tf.float32))

    def call(self, inputs):
        scaled_kernel = self.kernel * self.runtime_scale
        # Replicate Conv2D's internal logic with scaled kernel
        outputs = tf.nn.conv2d(inputs, scaled_kernel, self.strides, ...)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, ...)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
```

**Why This Was Challenging:**

1. **Keras Subclassing Overhead:** Subclassing `tf.keras.layers.Conv2D` meant inheriting its internal variable management, which interacts with TF2's graph compilation differently than manual TF1 graph construction

2. **Variance Propagation Issues:** Stacking 5-7 WScaleConv2D layers in the Encoder led to progressive variance decay:
   - Input std ≈ 1.0 → Output std ≈ 0.2-0.4 (target: ~1.0)
   - Even with `gain=sqrt(2.0)` (appropriate for ReLU-family activations)
   - Adding Pixel Normalization worsened this to std ≈ 0.004

3. **Decoder Output Range:** Unlike OG DFL where decoder outputs naturally stayed in [0,1] range (as if post-sigmoid), the TF2 refactor's decoder logits exhibited:
   - Too constrained: [-0.4, 0.8] with low gain → poor contrast after sigmoid
   - Exploding: thousands with high gain → saturation and noise
   - Attempts to tune output layer gain (0.1, 1.0, sqrt(2.0), 2.0) all failed

4. **Subtle Implementation Differences:** The way TF2/Keras compiles custom layers into graph operations appears to create subtly different execution paths compared to TF1's manual graph construction, affecting training dynamics in ways that are numerically stable but behaviorally distinct.

### Learning Outcome

This challenge provided deep insight into:
- How initialization and scaling affect gradient flow through networks
- The importance of variance preservation across layers
- Why "mathematically equivalent" implementations can exhibit different training behavior
- Framework-specific internals that impact training dynamics beyond just numerical correctness

## Technical Achievements

### VRAM Optimization Results

The migration to native TF2/Keras architecture achieved substantial memory reduction even without successfully implementing gradient checkpointing:

| Configuration | Original TF1 | TF2 Refactor | Improvement |
|--------------|--------------|--------------|-------------|
| Max Resolution (BS=16) | 412x412 | 704x704 | 2.9x pixel count |
| VRAM Usage (412px, BS=16) | ~20GB | ~12GB | 40% reduction |
| VRAM Usage (704px, BS=16) | OOM | ~24GB | Previously impossible |

*Hardware: NVIDIA RTX 5090 (32GB VRAM)*

![VRAM Comparison - TF1 at 412px](screenshots/vram_usage/vram_tf1_352px_bs16.png)
*Original TF1 implementation at 412x412 resolution, batch size 16*

![VRAM Comparison - TF2 at 704px](screenshots/vram_usage/vram_tf2_704px_bs16.png)
*TF2 refactor at 704x704 resolution, batch size 16 - nearly 3x the pixel count*

The ability to train at 704x704 resolution represented a fundamental shift in what was possible with the available hardware, even though training quality issues prevented these configurations from being practical for production use.

### Framework Modernization

Successfully migrated the entire training pipeline from TensorFlow 1.x paradigms to native TensorFlow 2.x:

**Training Loop**
- Replaced session-based execution with `tf.GradientTape` for automatic differentiation
- Implemented `@tf.function` graph compilation for the distributed training step
- Integrated `tf.distribute.MirroredStrategy` for multi-GPU training
- Moved from manual session management to eager execution model

**Model Architecture**
- Refactored 20+ custom layers from `LayerBase` to `tf.keras.layers.Layer`
- Migrated architectural components (`Encoder`, `Inter`, `Decoder`) to `tf.keras.Model`
- Implemented proper `build(input_shape)` and `call(inputs, training=False)` methods
- Used `self.add_weight()` for variable creation within layers

**Data Pipeline**
- Wrapped existing Python generators with `tf.data.Dataset.from_generator`
- Defined `output_signature` for type safety
- Integrated with `strategy.experimental_distribute_dataset` for distributed training

**Checkpointing**
- Replaced custom TF1 variable saving with `tf.train.Checkpoint`
- Implemented `tf.train.CheckpointManager` for checkpoint rotation
- Separated model options/history (pickled) from TensorFlow graph variables

### Custom Layer Engineering

Developed Keras-compatible implementations of DeepFaceLab's custom training techniques:

**WScaleConv2D and WScaleDense**
- Subclassed `tf.keras.layers.Conv2D` and `tf.keras.layers.Dense`
- Implemented equalized learning rate (weight scaling) via runtime kernel multiplication
- Kernels initialized from N(0,1), then scaled by `gain / sqrt(fan_in)` during forward pass
- Configurable gain parameter for variance control

**Architecture Components**
- Dependency injection pattern for layer composability
- Centralized layer class selection in top-level model file
- Maintained compatibility with multiple model variants (LIAE-UDT, DF, etc.)

## Technical Challenges

### Primary Challenge: Training Quality Disparity

Despite achieving numerical stability and correct gradient flow, the refactored implementation failed to replicate the original TF1 version's training characteristics:

**Observed Behavior**
- Slow visual convergence compared to original (1500+ iterations with minimal recognizable features)
- Preview outputs remained flat gray/orange/brown throughout early training
- Loss values decreased numerically but didn't correlate with visual improvement
- Decoder output logits exhibited suboptimal range characteristics

**Visual Comparison**

The following screenshots demonstrate the training quality disparity even at very high iteration counts:

![TF1 Baseline at 25,355 iterations](screenshots/training_comparison/tf1_baseline_iter_25355.png)
*Original TF1 implementation at iteration 25,355 - clear facial features and good reconstruction quality*

![TF2 Refactor at 26,549 iterations](screenshots/training_comparison/tf2_refactor_iter_26549.png)
*TF2 refactor at iteration 26,549 - visible reconstruction but lower detail and contrast despite similar iteration count*

These preview comparisons illustrate the core challenge: while the TF2 implementation was numerically stable and training progressed (loss decreased), the visual quality and learning speed did not match the original TF1 implementation's characteristics.

**Root Cause Analysis**

Investigation revealed fundamental differences in signal propagation:

1. **Variance Collapse Through Encoder Stack**
   - With Pixel Normalization enabled: Output std collapsed from ~1.0 to ~0.004
   - With Pixel Normalization disabled: Output std reached only ~0.2-0.4 (target: ~1.0)
   - Chain of 5 WScaleConv2D layers showed progressive variance decay despite gain=sqrt(2.0)

2. **Decoder Output Logit Range Issues**
   - Original TF1: Logits naturally constrained to [0,1] range (effectively post-sigmoid)
   - TF2 Refactor: Logits either too constrained ([-0.4, 0.8]) or exploding (thousands)
   - Attempts to tune output layer gain (0.1, 1.0, sqrt(2.0), 2.0) failed to achieve optimal range

3. **WScale Implementation Subtleties**
   - Keras subclassing approach mathematically correct but behaviorally different
   - Suspected incompatibility between custom variable handling and TF2 graph linking
   - Original TF1's manual graph construction exhibited properties not replicated in TF2

### Gradient Checkpointing Investigation

Extensive effort invested in implementing `tf.recompute_grad` yielded disappointing results:

**Implementation Challenges**
- `ValueError` with keyword arguments in custom gradient functions (resolved via positional args)
- `IndexError` from captured loop variables in closures (resolved with `functools.partial`)
- Complex integration with `@tf.function` and distribution strategy

**Performance Characteristics (TensorFlow 2.6.0)**
- No observable VRAM reduction when enabled
- 2x increase in GPU step time due to forward pass recomputation
- Eventually disabled globally for stability and performance

The reasons for gradient checkpointing's ineffectiveness remain unclear - potentially related to TF version, model architecture specifics, or hardware interaction patterns.

### Debugging Journey Highlights

**Import Resolution Errors (Significant Challenge)**

One of the most time-consuming debugging challenges involved resolving a cascade of import errors that emerged during the TF1 to TF2 migration:

- **Circular Dependencies:** TF2's eager execution model and stricter module initialization requirements exposed circular import patterns that worked (by accident) in TF1's lazy graph construction
- **Module Path Resolution:** Differences in how TF1 and TF2 handle Python module paths and relative imports caused numerous `ModuleNotFoundError` and `ImportError` exceptions
- **Systematic Isolation Required:** Each import error had to be debugged individually by:
  * Commenting out entire modules to isolate which imports were failing
  * Tracing import chains through multiple levels of nested dependencies
  * Testing import statements in isolation via Python REPL
  * Restructuring import order and using explicit imports vs. wildcard imports
- **Resolution Strategy:** Methodically refactored import structure through:
  * Explicit module path declarations
  * Breaking circular dependencies by moving shared utilities
  * Initializing modules in specific order within package `__init__.py` files
  * Converting relative imports to absolute where appropriate

This debugging process took several days of dedicated effort as each fix would often reveal new downstream import conflicts. The experience demonstrated the importance of understanding Python's import system deeply when working with complex, interconnected codebases.

**Convolution Depth Mismatch (`ValueError: Depth of input (X) is not a multiple of input depth of filter (Y)`)**
- Most time-consuming issue encountered
- Occurred with custom `Conv2D` layer despite correct Python-level shapes
- Root cause: Incompatibility between TF1-style variable handling and TF2 graph op linking
- Resolution: Switch to standard `tf.keras.layers.Conv2D` (sacrificing wscale feature)

**Distribution Strategy Integration**
- Moving `@tf.function` from model's `train_step` to outer `distributed_train_step` resolved merge_call errors
- Learning rate dropout feature disabled due to conflicts with conditional gradient application in strategy scope

**Loss Function Imports**
- Case sensitivity issues (`MSSimLoss` vs `MsSsimLoss`)
- Transitioning from functional loss helpers to `tf.keras.losses.Loss` subclasses
- Ensuring proper reduction for distributed training

## Comparative Analysis: TF1 vs TF2 Behavior

To understand the disparity, the original TF1 implementation was instrumented with debugging outputs:

**Original TF1 Decoder Outputs (1200-1300 iterations)**
- Image outputs: min≈0.0, max≈1.0, mean≈0.35-0.50
- Mask outputs: min=0.0, max=1.0, mean≈0.30-0.39
- Outputs behaved as if already sigmoid-activated despite no explicit sigmoid in Decoder module

**TF2 Refactor Decoder Outputs (1200-1300 iterations)**
- Image outputs: Highly variable depending on configuration
- With small output gain: min≈-0.4, max≈0.8 (sigmoid→~0.4-0.7, low contrast)
- With higher output gain: min/max in thousands (sigmoid→saturation, noisy output)
- Never achieved the stable [0,1]-like range of original implementation

This fundamental difference in Decoder output characteristics directly explained the poor visual feedback during training. The original implementation's specific combination of custom TF1 layers and graph construction produced self-limiting behavior that was not straightforwardly replicated in the Keras subclassing paradigm.

## Project Evolution and Decision Points

### Development Timeline

1. **Conceptualization Phase (February 2025):** 
   - Initial investigation into gradient checkpointing as primary VRAM optimization technique
   - Research into TensorFlow 2.x's `tf.recompute_grad` capabilities
   - Analysis of original DFL codebase architecture and dependencies
   - Feasibility assessment for framework migration

2. **Planning Phase (March 2025):** 
   - Comprehensive architecture design mapping TF1 components to TF2 equivalents
   - Identified critical components requiring refactoring: layers, models, training loop, data pipeline
   - Planned phased migration strategy to maintain functionality throughout development
   - Documented expected challenges and mitigation approaches

3. **Development & Debugging Phase (April 2025):** 
   - Week 1-2: Core layer migration to Keras architecture
     * Refactored base layers (Conv2D, Dense, Downscale, Upscale)
     * Implemented WScale variants with equalized learning rate
     * Built architectural components (Encoder, Inter, Decoder)
   - Week 2-3: Training loop and distribution strategy implementation
     * Converted session-based training to GradientTape
     * Integrated MirroredStrategy for multi-GPU support
     * Implemented checkpointing with tf.train.Checkpoint
   - Week 3-4: Systematic debugging of complex technical issues
     * **Import Resolution Challenges:** Spent significant time isolating circular dependencies and module path issues one by one
     * Resolved layer compatibility and shape mismatch errors
     * Fixed distribution strategy integration conflicts
     * Debugged numerical stability issues (NaNs, gradient explosion)

4. **Analysis & Extensive Tuning Phase (Late April - May 2025):** 
   - Comparative benchmarking with original TF1 implementation revealed quality gap
   - Instrumentation of both codebases to identify behavioral differences:
     * Added `tf.print` statements to track activation ranges throughout encoder/decoder
     * Monitored variance propagation at each layer boundary
     * Logged decoder output logit distributions pre- and post-sigmoid
   - Attempted numerous fixes for variance decay:
     * Experimented with different gain values (0.1, 1.0, sqrt(2.0), 2.0)
     * Toggled Pixel Normalization on/off in encoder
     * Tried alternative normalization techniques (batch norm, instance norm)
     * Adjusted dense layer dimensions in Inter block
     * Modified residual block structures
   - Tested decoder output layer configurations:
     * Standard Keras Conv2D vs WScaleConv2D for final layers
     * Different activation functions (tanh, linear, sigmoid at various stages)
     * Explicit sigmoid placement (within decoder vs. in loss calculation)
   - Profiled both implementations to rule out numerical instability
   - Documentation of findings and persistent training quality disparities

5. **Decision Point (June 2025):** 
   - After 2-3 weeks of intensive debugging and tuning attempts, recognized that achieving training quality parity would require disproportionate time investment
   - Core issue: The interaction between Keras subclassing, TF2 graph compilation, and wscale implementation created subtle behavioral differences that were difficult to isolate and correct
   - VRAM optimization goal was achieved (60% reduction), validating the technical approach
   - Training quality gap, while frustrating, provided valuable learning about deep learning framework internals
   - **Strategic pivot:** Focus efforts on optimizing proven-stable OG DFL TF1 codebase:
     * Implementing robust FP16 mixed-precision training for additional VRAM savings
     * Enhancing XSeg workflow for better mask quality
     * Improving merger quality-of-life features and preview generation
     * Leveraging new hardware (RTX 5090, 32GB) with stable TF1 for higher-resolution training
   - TF2 refactor retained as successful technical demonstration, comprehensive learning experience, and portfolio piece

### Strategic Pivot Rationale

After achieving significant VRAM improvements but failing to match training quality, development effort shifted back to the original TF1 MVE codebase with a focus on:

- Implementing stable FP16 training in TF1 for additional VRAM savings
- Enhancing XSeg workflow for better masking
- Improving merger quality-of-life features
- Leveraging new hardware (RTX 5090) for higher-resolution training with proven-stable TF1 implementation

The TF2 refactor successfully demonstrated the feasibility of major VRAM reductions and validated modernization approaches, but the training quality gap made it impractical for production use.

## Repository Structure

```
deepfacelab-tf2-refactor/
├── core/
│   └── leras/                    # Core neural network framework
│       ├── nn.py                 # TF2 initialization, GPU config
│       ├── layers/               # Custom Keras layers
│       │   ├── WScaleConv2D.py  # Custom weight-scaled convolution
│       │   ├── WScaleDense.py   # Custom weight-scaled dense
│       │   ├── Downscale.py     # Downsampling layer
│       │   ├── Upscale.py       # Upsampling layer (depth_to_space)
│       │   ├── ResidualBlock.py # Residual connection block
│       │   └── ...              # Other layer implementations
│       ├── archis/               # Architecture components
│       │   ├── Encoder.py       # Encoder network (tf.keras.Model)
│       │   ├── Inter.py         # Intermediate/latent space network
│       │   └── Decoder.py       # Decoder network
│       ├── losses/               # Loss functions
│       │   ├── DssimLoss.py     # DSSIM loss (tf.keras.losses.Loss)
│       │   └── MsSsimLoss.py    # Multi-scale SSIM loss
│       └── optimizers/           # Custom optimizers
│           └── RMSprop.py       # DFL RMSprop with LR dropout
├── models/
│   ├── ModelBase.py             # Base model utilities
│   └── Model_SAEHD/
│       └── Model.py             # Main SAEHD training model (TF2)
├── mainscripts/
│   └── Trainer.py               # Main training loop (TF2 + Strategy)
├── samplelib/                   # Data pipeline components
│   ├── Sample.py
│   ├── SampleProcessor.py
│   └── SampleGeneratorFace.py
└── main.py                      # Entry point
```

## Key Implementation Details

### Dependency Injection Pattern

Architectural components receive layer classes as constructor arguments rather than importing them directly:

```python
# In Model_SAEHD/Model.py
BaseConv2D_cls = WScaleConv2D  # or tf.keras.layers.Conv2D
BaseDense_cls = WScaleDense

self.encoder = Encoder(
    Conv2D_cls=BaseConv2D_cls,
    Downscale_cls=Downscale,
    # ... other injected classes
)
```

This centralized layer selection enables rapid experimentation with different implementations during debugging.

### WScale Implementation Example

```python
class WScaleConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, gain=math.sqrt(2.0), **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer=Zeros(),
            **kwargs
        )
        self.gain = gain
        self.runtime_scale = None

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.prod(self.kernel.shape[:-1])  # kh * kw * in_channels
        self.runtime_scale = self.gain * tf.math.rsqrt(tf.cast(fan_in, tf.float32))

    def call(self, inputs):
        scaled_kernel = self.kernel * self.runtime_scale
        outputs = tf.nn.conv2d(inputs, scaled_kernel, ...)
        # ... bias and activation
        return outputs
```

### Training Loop Structure

```python
@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(model.train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

# Main loop
for iteration in range(target_iterations):
    batch_data = next(dist_iterator)
    losses = distributed_train_step(batch_data)
    # ... logging, checkpointing
```

## Performance Characteristics

**GPU Step Time**
- Original TF1: ~300-400ms (320px, BS=16)
- TF2 Refactor: ~100-250ms (comparable configuration)
- TF2 with recompute_grad: ~500ms+ (not beneficial)

**Total Iteration Time**
- Dominated by CPU-bound data pipeline (~1.4-2.0s per iteration)
- Profiling identified mask generation (blur/dilate) as primary bottleneck
- GPU improvements offset by data pipeline limitations

## Skills Demonstrated

- Deep learning framework internals (TensorFlow 1.x/2.x)
- Custom layer development and gradient computation
- Distributed training strategy implementation
- Systematic debugging of complex DL systems
- Performance profiling and optimization
- Variance propagation analysis in deep networks
- Software architecture patterns (dependency injection, strategy pattern)
- Technical documentation and project analysis

## Running the Code

**Note:** This codebase is provided for educational and portfolio purposes. It is not recommended for production use due to the training quality issues described above.

### Requirements
- Python 3.6.8
- TensorFlow 2.6.0
- NVIDIA GPU with CUDA support
- Additional dependencies typically bundled with DeepFaceLab distributions

### Basic Usage

Run the training batch file:
```bash
6) train SAEHD.bat
```

The batch file will launch the training script and present interactive prompts for model configuration including resolution, architecture type, loss functions, batch size, and other training parameters. Configuration options are maintained from the original DFL implementation.

## Systematic Debugging Methodology

This project required developing and applying systematic approaches to debug complex deep learning systems. The methodology evolved through necessity:

### Component Isolation Strategy

1. **Layer-by-Layer Verification**
   - Built test harnesses for individual layers (Conv2D, Dense, Upscale, Downscale)
   - Created minimal reproducible examples with known input/output pairs
   - Compared TF2 layer outputs against OG TF1 equivalents with identical inputs
   - Verified both forward pass outputs and gradient computations

2. **Incremental Integration Testing**
   - Started with simplest architecture (single encoder downscale layer)
   - Progressively added complexity: residual blocks, full encoder, inter, decoder
   - Validated each addition by comparing activation statistics with OG DFL
   - Identified where variance propagation first diverged from expected behavior

### Instrumentation Techniques

**TensorFlow Debugging Tools:**
```python
# Variance monitoring
encoded = self.encoder(inputs)
encoded = tf.debugging.check_numerics(encoded, "Encoder output")
encoded = tf.print("Encoder std:", tf.math.reduce_std(encoded), output_stream=sys.stdout)

# Range verification
logits = self.decoder(latent_code)
tf.print("Logits - min:", tf.reduce_min(logits), "max:", tf.reduce_max(logits), "mean:", tf.reduce_mean(logits))
```

**Comparative Analysis:**
- Instrumented both TF1 and TF2 implementations with identical debug outputs
- Ran both versions with same random seeds and input data
- Compared activation ranges, variance, and gradient magnitudes at each layer
- Identified Decoder output range as critical divergence point

**Profiling:**
- Used `cProfile` and `tf.profiler` to identify performance bottlenecks
- Discovered CPU-bound data pipeline was limiting despite GPU optimizations
- Found mask generation (blur/dilate) consumed significant time

### Hypothesis-Driven Debugging

**Variance Collapse Investigation:**
1. **Hypothesis:** Pixel Normalization was causing variance decay
   - **Test:** Disabled pixel norm, monitored encoder output std
   - **Result:** Std improved from 0.004 to 0.2-0.4, but still below target of ~1.0
   - **Conclusion:** Pixel norm was a contributing factor but not the root cause

2. **Hypothesis:** Gain parameter needed tuning for TF2/Keras context
   - **Test:** Experimented with gain values from 0.1 to 2.0
   - **Result:** No gain value produced both stable variance AND good output range
   - **Conclusion:** Issue was more fundamental than parameter tuning

3. **Hypothesis:** TF2 graph compilation of WScale layers behaved differently
   - **Test:** Switched to standard Keras Conv2D (no wscale)
   - **Result:** More stable training but much slower visual convergence
   - **Conclusion:** Wscale was essential for OG DFL's training speed

**Decoder Output Investigation:**
1. **Hypothesis:** Missing sigmoid activation in decoder
   - **Test:** Added explicit sigmoid in decoder forward method
   - **Result:** Training diverged immediately (double sigmoid effect)
   - **Conclusion:** Loss calculation was already applying sigmoid

2. **Hypothesis:** Final Conv2D layer needed different initialization
   - **Test:** Used standard Keras Conv2D with default init for output layers
   - **Result:** Better output range but lost wscale benefits
   - **Conclusion:** Trade-off between output stability and training dynamics

### Documentation and Version Control Practices

- Maintained detailed change logs of each experimental modification
- Used Git branches for major experimental directions
- Documented negative results as thoroughly as positive ones
- Created comparison matrices of different configuration attempts
- Preserved instrumented versions of both TF1 and TF2 for future reference

### Key Debugging Insights

1. **Never Assume Equivalence:** "Mathematically equivalent" implementations can behave differently in practice due to framework internals

2. **Instrument Early:** Add debugging outputs from the start rather than retrofitting them when problems emerge

3. **Comparative Baselines Essential:** Having a working reference implementation (OG TF1) was invaluable for identifying where behavior diverged

4. **Document Everything:** Especially negative results - knowing what doesn't work is as valuable as knowing what does

5. **Know When to Pivot:** After exhausting reasonable debugging approaches, recognizing diminishing returns is important

## Key Technical Learnings from This Project

This refactoring effort provided deep insights into deep learning systems that extend far beyond framework-specific APIs:

### 1. Understanding Gradient Computation and Backpropagation

**Before This Project:**
- Conceptual understanding of backpropagation from academic courses
- Used automatic differentiation as a "black box" tool
- Focused on loss values and convergence curves

**After This Project:**
- **Deep understanding of gradient flow:** How gradients propagate through custom layers, how scaling factors affect gradient magnitudes, why some layers receive larger updates than others
- **Variance preservation importance:** Learned that maintaining ~unit variance through the network isn't just good practice - it's critical for stable training
- **The role of initialization:** Understood viscerally why He/Xavier initialization matters, and how equalized learning rate (wscale) provides an alternative approach
- **Gradient clipping mechanics:** Implemented and understood why gradient clipping by norm prevents exploding gradients in practice

**Concrete Example:**
When the encoder output variance collapsed to 0.004, I learned that:
- Low variance → small gradients during backprop → slow learning in early layers
- This cascades: encoder learns slowly → latent codes don't evolve → decoder has nothing useful to work with
- Solution requires addressing the full propagation chain, not just individual layer tuning

### 2. Framework Internals: TF1 vs TF2

**TensorFlow 1.x Graph Construction:**
- Static computation graph defined upfront
- Manual session management and variable scoping
- Explicit placeholder→operation→session.run() pipeline
- Custom layers: `build_weights()` creates variables in graph, `forward()` defines ops
- **Key insight:** Graph construction order and op linking can affect execution in subtle ways

**TensorFlow 2.x / Keras Paradigm:**
- Eager execution by default, graph compilation via `@tf.function`
- Automatic differentiation with `GradientTape`
- Keras layers: `build()` for weight creation, `call()` for computation
- Subclassing introduces framework-managed variable tracking
- **Key insight:** Keras's internal variable management interacts with custom scaling logic in ways that differ from manual TF1 graph construction

**Critical Difference Discovered:**
In TF1, doing `weight * scale_constant` in a layer's forward method creates a specific graph op that flows into `tf.nn.conv2d` in a certain way. In TF2/Keras, subclassing `Conv2D` and modifying `self.kernel * self.runtime_scale` in `call()` goes through Keras's internal call stack, affecting how the operation is compiled by `@tf.function`. This subtle difference impacted training dynamics.

### 3. Why "Mathematically Equivalent" ≠ "Behaviorally Equivalent"

One of the most important lessons:

**Numerical Correctness is Necessary But Not Sufficient**
- Both implementations computed correct gradients (verified)
- Loss functions matched exactly
- Forward pass outputs were numerically stable
- Yet training dynamics diverged significantly

**Factors Beyond Math:**
- **Operation ordering:** The sequence in which ops execute can affect floating-point precision accumulation
- **Graph compilation strategies:** How `@tf.function` compiles custom layers affects performance and behavior
- **Memory layout:** TF1 vs TF2 may layout tensors in memory differently, affecting cache hits and parallel execution
- **Framework internals:** How Keras manages variables, applies regularization, and tracks updates differs from manual TF1 approaches

**Real-World Implication:**
A "perfect" refactor on paper (same architecture, same math, same hyperparameters) can still fail in practice due to framework-level differences. This explains why many ML frameworks have version-specific quirks and why reproducibility across frameworks is challenging.

### 4. The Importance of Training Dynamics

**What I Learned:**
Training neural networks isn't just about reaching a minimum - it's about *how* you traverse the loss landscape:

- **Convergence speed matters:** OG DFL showed recognizable faces within 500-1000 iterations. TF2 refactor took 10x longer. For researchers/users, this is the difference between usable and unusable
- **Optimization path matters:** Two implementations can reach similar final losses but via different paths, resulting in different learned features
- **Early training is critical:** The first few hundred iterations set the "tone" for what features the network learns. Poor early dynamics can lead to local minima that are hard to escape
- **Empirical tuning is cumulative:** OG DFL's training behavior resulted from years of community tuning. Replicating this isn't just about copying hyperparameters - the entire system interaction matters

### 5. Software Engineering in ML Context

**Dependency Injection Pattern:**
- Originally seemed like unnecessary abstraction
- Proved invaluable for rapid experimentation (swapping WScaleConv2D ↔ standard Conv2D)
- Enabled testing hypotheses quickly without deep code changes
- **Lesson:** ML code benefits from traditional software engineering patterns, even though it's "just research code"

**Testing and Validation:**
- Unit tests for individual layers were essential
- Integration tests with known input/output pairs caught subtle bugs
- **Lesson:** ML debugging is harder than traditional debugging because correct execution doesn't guarantee correct learning

**Version Control Discipline:**
- Branching strategies for experimental directions
- Detailed commit messages documenting hypothesis being tested
- **Lesson:** Treat ML experiments like any other software development - document, version, and track changes systematically

### 6. Performance Optimization is Multi-Dimensional

Achieved 60% VRAM reduction and 2-3x faster GPU step time, yet:
- Total iteration time barely improved (CPU-bound data pipeline)
- Gradient checkpointing failed to provide VRAM benefits despite implementation effort
- **Lesson:** Optimize the bottleneck, not what's easy to optimize. Profiling is essential.

### 7. Knowing When to Pivot

Perhaps the most valuable meta-lesson:
- After 2-3 weeks of intensive debugging with diminishing returns, recognized that perfect parity might not be achievable within reasonable time
- The refactoring achieved its primary goal (VRAM reduction) and demonstrated technical competence
- Shifting focus to FP16 optimization of proven-stable OG DFL was the practical choice
- **Lesson:** Technical projects need clear success criteria and exit strategies. Not every problem needs to be solved completely.

### 8. Deep Learning is Still a Young Field

This experience reinforced that:
- Despite mature APIs, deep learning frameworks still have rough edges
- Training dynamics are often poorly understood theoretically
- Empirical tuning still dominates over principled approaches
- "Best practices" may be framework-specific or even version-specific
- There's still much to learn about why neural networks train the way they do

## Lessons Learned

1. **Framework Migration Complexity:** Even mathematically equivalent implementations can exhibit different training dynamics due to subtle differences in operation ordering, variable initialization, or graph construction.

2. **Debugging Deep Learning Systems:** Requires systematic isolation of components, extensive instrumentation, and comparative analysis with known-good baselines.

3. **Custom Layer Development:** Subclassing modern frameworks requires careful attention to internal implementation details that may not be apparent from API documentation alone.

4. **VRAM Optimization:** Significant memory reductions possible through framework-level changes even without specialized techniques like gradient checkpointing.

5. **Production Readiness vs. Technical Achievement:** A successful refactoring requires not just numerical correctness but behavioral equivalence across all relevant metrics.

## Future Directions

Potential paths for continued investigation:

- Test with newer TensorFlow versions (2.10+) for improved gradient checkpointing
- Explore alternative variance normalization techniques (Group Norm, Layer Norm variants)
- Investigate mixed-precision training implementation in TF2 context
- Consider PyTorch migration for comparison of framework differences
- Detailed profiling of original TF1 graph construction to identify subtle differences

## License

This refactoring work maintains the original DeepFaceLab license:

**GNU General Public License v3.0 (GPLv3)**

The original DeepFaceLab framework and this derivative refactoring work are licensed under the GNU General Public License version 3. This is a free, copyleft license that ensures:

- Freedom to use, study, modify, and distribute the software
- Any modifications or derivative works must also be distributed under GPLv3
- Source code must be made available when distributing the software
- Changes made to the code must be documented

See the full license text at: https://www.gnu.org/licenses/gpl-3.0.html

Original DeepFaceLab: Copyright (C) 2018-2020 Ivan Petrov, Petr Yaroshenko, and contributors

---

**Project Timeline:** February 2025 - June 2025  
**Hardware:** NVIDIA RTX 5090 (32GB), AMD Ryzen 9950X3D  
**Development Approach:** Iterative refactoring with systematic debugging and comparative analysis
