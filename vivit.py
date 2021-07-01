# %%
import tensorflow as tf
from tensorflow import nn, einsum
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.layers import TimeDistributed
# import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(tf.keras.layers.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer, self).__init__()
        self.layers = []
        self.norm = LayerNormalization()
        for _ in range(depth):
            self.layers.append([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ])

    def call(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_classes, num_frames, 
        batch_size=32, dim = 192, depth = 4, heads = 3, pool = 'cls', 
        in_channels = 3, dim_head = 64, dropout = 0.,
        emb_dropout = 0., scale_dim = 4, ):
        super(ViViT, self).__init__()
        
        assert pool in {'cls', 'mean', 'time'}, \
        'pool type must be either cls (cls token), mean (mean pooling), \
            or time (time distributed)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = tf.keras.Sequential([
            Input(shape=(num_frames, in_channels, image_size, image_size,), 
                batch_size=batch_size),
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            Dense(dim),
        ])

        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, num_frames, num_patches + 1, dim))
        self.space_token = self.add_weight("space_token", shape=(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = self.add_weight("temp_token", shape=(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = tf.keras.Sequential([
            LayerNormalization(),
            Dense(num_classes)
        ])

    def call(self, inputs):
        x = self.to_patch_embedding(inputs)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = tf.concat((cls_space_tokens, x), axis=2)
        print("now x:", x.shape)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = tf.concat((cls_temporal_tokens, x), axis=1)

        x = self.temporal_transformer(x)
        
        # if mean, average across time points
        if self.pool == 'mean':
            x = tf.math.reduce_mean(x, axis=1) 
            x = self.mlp_head(x)
        # elif classification, use the first (classification) token
        elif self.pool == 'cls': 
            x = x[:, 0]
            x = self.mlp_head(x)
        # elif time, use TimeDistributed layer to apply MLP to each time step
        elif self.pool == 'time':
            # skip the first (classification) token
            x = TimeDistributed(self.mlp_head)(x[:, 1:, :])
        return x
    
if __name__ == "__main__":
    
    num_samples = 10000
    batch_size = 32
    X_pos = np.random.normal(loc=1., scale=1., 
    size=(int(num_samples/2), 10, 6, 36, 36))
    y_pos = np.ones(shape=(int(num_samples/2), 10, 2), dtype=np.float)

    X_neg = np.random.normal(loc=0., scale=1., 
    size=(int(num_samples/2), 10, 6, 36, 36))
    y_neg = np.zeros(shape=(int(num_samples/2), 10, 2), dtype=np.float)
    
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    print(X.shape, y.shape)

    model = ViViT(image_size=36, patch_size=12, in_channels=6, 
    num_classes=2, num_frames=10, dim=128, pool='time')
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.build(input_shape=(batch_size, *X.shape[1:]))
    tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
    print(model.summary())
    model = model.fit(X, y, batch_size=batch_size)
