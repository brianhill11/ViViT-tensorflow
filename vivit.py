# %%
import tensorflow as tf
from tensorflow import nn, einsum
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.layers import TimeDistributed, Reshape, Lambda
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
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_classes, num_frames, 
        batch_size=32, dim=192, depth=4, heads=3, pool='cls', 
        in_channels=3, dim_head=64, dropout=0.,
        emb_dropout=0., scale_dim=4, output_names=None):
        super(ViViT, self).__init__()
        
        # name of final layer output 
        self.output_names = output_names
        assert pool in {'cls', 'mean', 'time'}, \
        'pool type must be either cls (cls token), mean (mean pooling), \
            or time (time distributed)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = tf.keras.Sequential([
            Input(shape=(num_frames, image_size, image_size, in_channels), 
                batch_size=batch_size),
            Rearrange('b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
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
        if self.output_names:
            # name each output layer 
            output = [Lambda(lambda a: a[:, 1:, i], name=name)(x) for i, name in enumerate(self.output_names)]
            x = output
        return x

if __name__ == "__main__":

    # simulate example data
    num_samples = 800
    batch_size = 8
    num_frames = 10
    image_size = 36
    in_channels = 3
    X_pos = np.random.normal(loc=1., scale=1., 
    size=(int(num_samples/2), num_frames+1, image_size, image_size, in_channels))
    y_pos = np.ones(shape=(int(num_samples/2), num_frames, 2), dtype=np.float)

    X_neg = np.random.normal(loc=0., scale=1., 
    size=(int(num_samples/2), num_frames+1, image_size, image_size, in_channels))
    y_neg = np.zeros(shape=(int(num_samples/2), num_frames, 2), dtype=np.float)
    
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    print(X.shape, y.shape)

    y = {"dysub": y[:, :, 0], "drsub": y[:, :, 1]}

    model = ViViT(image_size=image_size, patch_size=image_size, in_channels=in_channels, 
    num_classes=2, num_frames=num_frames+1, dim=16, pool='time', batch_size=batch_size, output_names=["dysub", "drsub"])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.build(input_shape=(batch_size, num_frames+1, *X.shape[2:]))
    # tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
    print(model.summary())
    history = model.fit(X, y, batch_size=batch_size)

# %%
