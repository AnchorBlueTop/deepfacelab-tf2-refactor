from core.leras import nn
tf = nn.tf
from core.leras.layers import ResidualBlock

class RB(tf.keras.Model):
    def on_build(self, ch, kernel_size=3 ):
        self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
        self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

    def forward(self, inp):
        x = self.conv1(inp)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x


class LCA(tf.keras.Modele):
    def on_build(self, ch):
        self.conv1 = nn.Conv2D( ch, ch, kernel_size=1, padding='VALID')

    def forward(self, inp):
        x = inp
        x = tf.reduce_mean(x, axis=nn.conv2d_spatial_axes, keepdims=True)
        x = self.conv1(x)
        x = tf.nn.sigmoid(x)
        x = inp * x
        return x

class RAFG(tf.keras.Model):
    def on_build(self, ch):
        self.rb1 = RB(ch)
        self.lca1 = LCA(ch)
        self.rb2 = RB(ch)
        self.lca2 = LCA(ch)
        self.rb3 = RB(ch)
        self.lca3 = LCA(ch)

        self.ab_conv = nn.Conv2D( ch*3, ch, kernel_size=1, padding='VALID')
        self.ab_lca = LCA(ch)
        self.fb_conv = nn.Conv2D( ch*4, ch, kernel_size=1, padding='VALID')

    def forward(self, inp):
        x = inp

        rb1 = self.rb1(x)
        lca1 = self.lca1(rb1)

        x = x+rb1

        rb2 = self.rb2(x)
        lca2 = self.lca2(rb2)

        x = x+rb2

        rb3 = self.rb3(x)
        lca3 = self.lca3(rb3)

        lca = tf.concat([lca1,lca2,lca3], axis=nn.conv2d_ch_axis)
        lca = self.ab_conv(lca)
        lca = self.ab_lca(lca)

        rb  = tf.concat([inp,rb1,rb2,rb2], axis=nn.conv2d_ch_axis)
        rb = self.fb_conv(rb)

        return rb+lca, lca


class HRAN(tf.keras.Model):
    """
    Hierarchical Residual Attention Network for Single Image Super-Resolution
    
    https://arxiv.org/pdf/2012.04578v1.pdf
    """
    
    
    def on_build(self, ch):
        self.rafg1 = RAFG(ch)
        self.rafg2 = RAFG(ch)
        self.rafg3 = RAFG(ch)

        self.ab_conv = nn.Conv2D( ch*3, ch, kernel_size=1, padding='VALID')
        self.ab_lca = LCA(ch)
        self.fb_conv = nn.Conv2D( ch*4, ch, kernel_size=1, padding='VALID')

    def forward(self, inp):
        x = inp

        rafg1, rafg1_lca = self.rafg1(x)

        rafg2, rafg2_lca = self.rafg2(x)
        rafg3, rafg3_lca = self.rafg3(x)
        
        rafg_lca = tf.concat([rafg1_lca,rafg2_lca,rafg3_lca], axis=nn.conv2d_ch_axis)
        rafg_lca = self.ab_conv(rafg_lca)
        rafg_lca = self.ab_lca(rafg_lca)
        
        rafg = tf.concat([x,rafg1,rafg2,rafg3], axis=nn.conv2d_ch_axis)
        rafg = self.fb_conv(rafg)
        
        x = x + rafg + rafg_lca        
        x = tf.nn.leaky_relu(x, 0.2)        

        return x
        
class DeepFakeArchi(tf.keras.Model):
    """
    resolution

    mod     None - default
            'quick'
    """
    def __init__(self, resolution, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''

        if mod is None:
            class Downscale(tf.keras.Model):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class DownscaleBlock(tf.keras.layers.Layer):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size) )
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x
                
            # Inherit from tf.keras.layers.Layer
            class Upscale(tf.keras.layers.Layer):
                """
                Upscales the input tensor using a Conv2D layer followed by depth_to_space
                (Pixel Shuffling). Compatible with Keras API.
                """
                def __init__(self, out_ch, kernel_size=3, activation=None, use_fp16=False, name=None, **kwargs):
                    """
                    Args:
                        out_ch: Integer, the number of output channels for the *final* upscaled layer.
                                The internal Conv2D will have 4*out_ch filters.
                        kernel_size: Integer, the kernel size for the internal Conv2D layer.
                        activation: Activation function to apply after the convolution.
                                    Can be a Keras activation function, a TensorFlow function,
                                    or None.
                        use_fp16: Boolean, whether to use float16 for the internal convolution.
                        name: Optional name for the layer.
                        **kwargs: Additional keyword arguments passed to the parent Layer.
                    """
                    super().__init__(name=name, **kwargs)
                    self.out_ch = out_ch
                    self.kernel_size = int(kernel_size)
                    self.activation = tf.keras.activations.get(activation) # Get activation function
                    self.use_fp16 = use_fp16
                    self.conv_dtype = tf.float16 if use_fp16 else tf.float32

                    # Internal Conv2D layer will be created in build()
                    self.conv1 = None

                def build(self, input_shape):
                    # Infer input channels from input_shape
                    if nn.data_format == "NHWC":
                        channel_axis = -1
                    else: # NCHW
                        channel_axis = 1
                    in_ch = input_shape[channel_axis]
                    if in_ch is None:
                        raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

                    # Create the internal Conv2D layer
                    # Note: We use the refactored nn.Conv2D assuming it's Keras-compatible
                    self.conv1 = nn.Conv2D(in_ch=in_ch, # Pass inferred in_ch
                                        out_ch=self.out_ch * 4, # Output 4x channels for depth_to_space
                                        kernel_size=self.kernel_size,
                                        padding='SAME', # Upscaling usually uses SAME padding
                                        dtype=self.conv_dtype, # Pass dtype for fp16 if needed
                                        name=f"{self.name}_conv2d" if self.name else None)

                    # Ensure Keras knows the layer is built
                    # Call build on the internal layer explicitly if needed, or rely on its first call
                    # super().build(input_shape) # Keras handles calling build on internal layers

                # Replaces 'forward' method
                def call(self, inputs):
                    if self.conv1 is None:
                        # Should have been built by Keras automatically, but explicitly call just in case
                        self.build(tf.shape(inputs))

                    x = self.conv1(inputs)

                    if self.activation is not None:
                        x = self.activation(x)

                    # Apply depth_to_space (Pixel Shuffling)
                    # block_size=2 means upscale by 2x height and 2x width
                    outputs = tf.nn.depth_to_space(x, block_size=2, data_format=nn.data_format)

                    return outputs

                # Optional: Implement get_config for Keras saving/loading compatibility
                def get_config(self):
                    config = super().get_config()
                    config.update({
                        'out_ch': self.out_ch,
                        'kernel_size': self.kernel_size,
                        'activation': tf.keras.activations.serialize(self.activation),
                        'use_fp16': self.use_fp16,
                        # 'name' and 'dtype' are handled by the base Layer get_config
                    })
                    return config

                def __str__(self):
                    return f"{self.__class__.__name__} (out_ch: {self.out_ch}, kernel: {self.kernel_size})"

            # Assign the refactored class back to nn.Upscale if needed for compatibility
            # nn.Upscale = Upscale

            class ResidualBlock(tf.keras.layers.Layers):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = tf.nn.leaky_relu(inp + x, 0.2)
                    return x

            class Encoder(tf.keras.Model):
                def __init__(self, in_ch, e_ch, **kwargs ):
                    self.in_ch = in_ch
                    self.e_ch = e_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4, kernel_size=5)

                def forward(self, inp):
                    return nn.flatten(self.down1(inp))

                def get_out_res(self, res):
                    return res // (2**4)

                def get_out_ch(self):
                    return self.e_ch * 8
            
            lowest_dense_res = resolution // 16
            
            if 'h' in opts:
                lowest_dense_res //= 2
                
            if 'd' in opts:
                lowest_dense_res //= 2

            class Inter(tf.keras.Model):
                def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
                    if 'u' in opts:
                        self.dense_norm = nn.DenseNorm()

                    self.dense1 = nn.Dense( in_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = inp
                    if 'u' in opts:
                        x = self.dense_norm(x)
                    x = self.dense1(x)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                    x = self.upscale1(x)
                    return x

                def get_out_res(self):
                    return lowest_dense_res * 2

                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(tf.keras.Model):
                def on_build(self, in_ch, d_ch, d_mask_ch ):
                    self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                    self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
        
                    self.res0 = HRAN(d_ch*8)
                    self.res1 = HRAN(d_ch*4)
                    self.res2 = HRAN(d_ch*2)
                    self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME')

                    self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                    self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                    self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                    self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')

                    if 'd' in opts:
                        self.out_conv1  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                        self.out_conv2  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                        self.out_conv3  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                    
                    if 'h' in opts and 'd' in opts:
                        self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*2, kernel_size=3)
                        self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                    elif 'h' in opts or 'd' in opts:
                        self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                    
                    if 'h' in opts or 'd' in opts:
                        self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME')
                    else:
                        self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')
                            
                    if 'h' in opts:
                        self.hran = HRAN(3, 64)
                        
                        
                        
                def forward(self, inp):
                    z = inp

                    x = self.upscale0(z)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)


                    if 'd' in opts:
                        x0 = tf.nn.sigmoid(self.out_conv(x))
                        x0 = nn.upsample2d(x0)
                        x1 = tf.nn.sigmoid(self.out_conv1(x))
                        x1 = nn.upsample2d(x1)
                        x2 = tf.nn.sigmoid(self.out_conv2(x))
                        x2 = nn.upsample2d(x2)
                        x3 = tf.nn.sigmoid(self.out_conv3(x))
                        x3 = nn.upsample2d(x3)
                        
                        tile_res = resolution // 2
                        if 'h' in opts:
                            tile_res //= 2
                            
                        if nn.data_format == "NHWC":
                            tile_cfg = ( 1, tile_res, tile_res, 1)
                        else:
                            tile_cfg = ( 1, 1, tile_res, tile_res)

                        z0 =  tf.concat ( ( tf.concat ( (  tf.ones ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )

                        z0 = tf.tile ( z0, tile_cfg )

                        z1 =  tf.concat ( ( tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.ones ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z1 = tf.tile ( z1, tile_cfg )

                        z2 =  tf.concat ( ( tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( (  tf.ones ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z2 = tf.tile ( z2, tile_cfg )

                        z3 =  tf.concat ( ( tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.ones ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z3 = tf.tile ( z3, tile_cfg )

                        x = x0*z0 + x1*z1 + x2*z2 + x3*z3
                    else:
                        x = tf.nn.sigmoid(self.out_conv(x))


                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)
                    
                    if 'h' in opts and 'd' in opts:
                        m = self.upscalem3(m)
                        m = self.upscalem4(m)
                    elif 'h' in opts or 'd' in opts:
                        m = self.upscalem3(m)                                            
                    m = tf.nn.sigmoid(self.out_convm(m))
                    
                    if 'h' in opts:
                        x = self.hran(x)
                        
                    return x, m

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

# nn.DeepFakeArchi = DeepFakeArchi