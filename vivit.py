# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.training import Model
from tensorflow_addons.layers import GELU
from tensorflow import nn, einsum
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.layers import TimeDistributed, Lambda
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange


def PreNorm(input_layer):
    return tf.keras.layers.LayerNormalization()(input_layer)

def FeedForward(input_layer, dim, hidden_dim, dropout=0.):
    output_layer = Dense(hidden_dim)(input_layer)
    output_layer = GELU()(output_layer)
    output_layer = Dropout(dropout)(output_layer)
    output_layer = Dense(dim)(output_layer)
    output_layer = Dropout(dropout)(output_layer)
    return output_layer

def Attention(input_layer, dim, heads=8, dim_head=64, dropout=0.):
    b, n, _, h = *input_layer.shape, heads
    inner_dim = dim_head * heads
    project_out = not (heads == 1 and dim_head == dim)
    scale = dim_head ** -0.5

    to_qkv = Dense(inner_dim * 3, use_bias = False)(input_layer)
    qkv = tf.split(to_qkv, num_or_size_splits=3, axis=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

    dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

    attn = tf.nn.softmax(dots, axis=-1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    if project_out:
        out = Dense(dim)(out)
        out = Dropout(dropout)(out)
    else: 
        out = nn.Identity(out)
    return out

def TransformerBlock(input_layer, dim, mlp_dim, heads=8, dim_head=64, dropout=0.):
    output_layer = PreNorm(input_layer)
    output_layer = Attention(output_layer, dim, heads=heads, dim_head=dim_head, 
        dropout=dropout)
    output_layer = PreNorm(output_layer)
    output_layer = FeedForward(output_layer, dim, hidden_dim=mlp_dim, 
        dropout=dropout)
    return output_layer

def Transformer(batch_size, num_patches, dim, depth, heads, dim_head, mlp_dim, 
    dropout=0., name="Transformer"):
    input_layer = Input(shape=(num_patches, dim), batch_size=batch_size)
    output_layer = input_layer
    for _ in range(depth):
        output_layer = TransformerBlock(output_layer, dim, mlp_dim, heads, 
            dim_head, dropout=dropout)
    output_layer = LayerNormalization()(output_layer)
    model = Model(input_layer, output_layer, name=name)
    return model

def ViViT(image_size, patch_size, num_classes, num_frames, 
        batch_size=32, dim=192, depth=4, heads=3, pool='cls', 
        in_channels=3, dim_head=64, dropout=0.,
        emb_dropout=0., scale_dim=4, 
        use_classification_token=True, use_temporal_token=True, 
        output_names=None):

    assert pool in {'cls', 'mean', 'time', 'none'}, \
        'pool type must be either cls (cls token), mean (mean pooling), \
            time (time distributed), or none (return embeddings)'

    assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

    num_patches = (image_size // patch_size) ** 2
    patch_dim = in_channels * patch_size ** 2
    mlp_dim = dim*scale_dim
    
    w_init = tf.random_normal_initializer()
    pos_embedding = tf.Variable(
            initial_value=w_init(
                shape=(1, num_frames, num_patches + 1, dim), 
                dtype="float32"),
            trainable=True,)

    # model definition
    input = Input(shape=(num_frames, image_size, image_size, in_channels), 
            batch_size=batch_size)
    output = Rearrange('b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)(input)
    output = Dense(dim)(output)

    b, t, n, _ = output.shape

    # create a classification token and add to list of patch tokens
    if use_classification_token:
        space_token = tf.Variable(
            initial_value=w_init(
                shape=(1, 1, dim), 
                dtype="float32"),
            trainable=True,)
        
        cls_space_tokens = repeat(space_token, '() n d -> b t n d', b = b, t=t)
        output = tf.concat((cls_space_tokens, output), axis=2)
    # add the positional embedding to each token
    output += pos_embedding[:, :, :(n + 1)]
    output = Dropout(emb_dropout, name="Dropout")(output)

    output = rearrange(output, 'b t n d -> (b t) n d')
    # spatial transformer 
    output = Transformer(output.shape[0], output.shape[1], output.shape[2], 
        depth, heads, dim_head, mlp_dim, 
        dropout=dropout, name="SpatialTransformer")(output)
    output = rearrange(output[:, 0], '(b t) ... -> b t ...', b=b)

    if use_temporal_token:
        temporal_token = tf.Variable(
                initial_value=w_init(
                    shape=(1, 1, dim), 
                    dtype="float32"),
                trainable=True,)
        
        cls_temporal_tokens = repeat(temporal_token, '() n d -> b n d', b=b)
        output = tf.concat((cls_temporal_tokens, output), axis=1)

    # temporal transformer
    output = Transformer(output.shape[0], output.shape[1], output.shape[2], 
        depth, heads, dim_head, mlp_dim, 
        dropout=dropout, name="TemporalTransformer")(output)
    
    # if mean, average across time points
    if pool == 'mean':
        output = tf.math.reduce_mean(output, axis=1) 
        output = LayerNormalization()(output)
        output = Dense(num_classes)(output)
    # elif classification, use the first (classification) token
    elif pool == 'cls': 
        output = output[:, 0]
        output = LayerNormalization()(output)
        output = Dense(num_classes)(output)
    # elif time, use TimeDistributed layer to apply MLP to each time step
    elif pool == 'time':
        # skip the first (classification) token
        # TODO: - can this skipping be handled by argument? 
        output = LayerNormalization()(output)
        output = TimeDistributed(Dense(num_classes))(output[:, 1:, :])
        if output_names:
            # name each output layer 
            temp_output = [Lambda(lambda a: a[:, 1:, i], name=name)(output) for i, name in enumerate(output_names)]
            output = temp_output
    # elif none, return token embeddings
    elif pool == 'none':
        # skip first (classification token)
        output = output[:, 1:, :]

    model = Model(inputs=input, outputs=output)
    return model



# simulate example data
num_samples = 800
batch_size = 8
num_frames = 30
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
num_classes=2, num_frames=num_frames+1, dim=16, pool='time', depth=1,
batch_size=batch_size, output_names=["dysub", "drsub"], 
use_classification_token=True, use_temporal_token=True,)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=False)
print(model.summary())
os.makedirs("checkpoints", exist_ok=True)
save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join("checkpoints", "checkpoint_epoch{epoch:02d}_model.hdf5"),
                                                            save_best_only=False, verbose=1)
history = model.fit(X, y, batch_size=batch_size, callbacks=[save_best_callback])
