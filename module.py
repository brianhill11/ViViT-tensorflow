import tensorflow as tf
from tensorflow import nn, einsum
from tensorflow.keras.layers import Dense, Dropout
from tensorflow_addons.layers import GELU
import numpy as np

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class Residual(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
    def call(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.fn = fn
    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            Dense(hidden_dim),
            GELU(),
            Dropout(dropout),
            Dense(dim),
            Dropout(dropout)
        ])
    def call(self, x):
        return self.net(x)

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = Dense(inner_dim * 3, use_bias = False)

        self.to_out = tf.keras.Sequential([
            Dense(dim),
            Dropout(dropout)
        ]) if project_out else nn.Identity()

    def call(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = tf.nn.softmax(dots, axis=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ReAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(ReAttention, self).__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = Dense(inner_dim * 3, use_bias = False)

        self.reattn_weights = tf.Variable(np.random.randn(heads, heads))

        self.reattn_norm = tf.keras.Sequential([
            Rearrange('b h i j -> b i j h'),
            tf.keras.layers.LayerNormalization(),
            Rearrange('b i j h -> b h i j')
        ])

        self.to_out = tf.keras.Sequential([
            Dense(dim),
            Dropout(dropout)
        ])

    def call(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class LeFF(tf.keras.layers.Layer):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super(LeFF, self).__init__()
        
        scale_dim = dim*scale
        self.up_proj = tf.keras.Sequential([
                                    Dense(scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
        ])
        
        self.depth_conv =  tf.keras.Sequential([nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
        ])
        
        self.down_proj = tf.keras.Sequential([
                                    Dense(dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    GELU(),
                                    Rearrange('b c n -> b n c')
        ])
        
    def call(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(LCAttention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = Dense(inner_dim * 3, use_bias = False)

        self.to_out = tf.keras.Sequential([
            Dense(dim),
            Dropout(dropout)
        ]) if project_out else nn.Identity()

    def call(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
        
        
        
        

