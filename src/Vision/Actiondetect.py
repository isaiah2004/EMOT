# import cv2
# import glob
import tensorflow as tf
# print(tf.__version__)

import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from IPython.display import clear_output
# from matplotlib.pyplot import imshow
# import pandas as pd
# from tensorflow.keras.layers import *
from tensorflow.keras.layers import *
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.metrics import *
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as k
# import tensorflow.python.keras.layers as layers
# import datetime
import os
# import csv
import pandas as pd
# import random
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
# from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Input,
    concatenate,
    Concatenate,
)
from keras.layers import (
    Dense,
    MaxPool2D,
    Flatten,
    GlobalAveragePooling2D,
    BatchNormalization,
    Layer,
    Add,
    UpSampling2D,
    Lambda,
    Multiply,
)
from keras.losses import (
    categorical_crossentropy,
    categorical_hinge,
    hinge,
    squared_hinge,
)
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

labels = [
    "Looking_Forward",
    "Raising_Hand",
    "Reading",
    "Sleeping",
    "Standing",
    "Turning_Around",
    "Writing",
]


# config#######################################################
def list_to_stack(xs):
    xs = tf.stack(xs, axis=1)
    s = tf.shape(xs)

    return xs


num_classes = 7
input_shape = (128, 128, 1)
patch_size = (2, 2)
dropout_rate = 0.03
num_heads = 8
embed_dim = 64
num_mlp = 256
qkv_bias = True
window_size = 2
shift_size = 1
image_dimension = 128

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 16
num_epochs = 100
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1
# config#######################################################


def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_prob": self.drop_prob,
            }
        )
        return config


class WindowAttention(layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        return_attention_scores=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.return_attention_scores = return_attention_scores
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )

        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1]
        )
        super().build(input_shape)

    def get_relative_position_index(self, window_height, window_width):
        x_x, y_y = tf.meshgrid(range(window_height), range(window_width))
        coords = tf.stack([y_y, x_x], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])

        x_x = (relative_coords[:, :, 0] + window_height - 1) * (2 * window_width - 1)
        y_y = relative_coords[:, :, 1] + window_width - 1
        relative_coords = tf.stack([x_x, y_y], axis=-1)

        return tf.reduce_sum(relative_coords, axis=-1)

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)

        if self.return_attention_scores:
            return x_qkv, attn
        else:
            return x_qkv

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "scale": self.scale,
            }
        )
        return config


class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        print("Value of dim:", self.dim)

        # Norm Layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        # MLP Layers
        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "dim": self.dim,
            "num_patch": self.num_patch,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "num_mlp": self.num_mlp,
        }
        return {**base_config, **config}

    # @keras.saving.register_keras_serializable(package="PatchExtract")


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size_y": self.patch_size_y,
                "patch_size_x": self.patch_size_x,
            }
        )
        return config


# @keras.saving.register_keras_serializable(package="MyLayers")
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        print(self.proj(patch) + self.pos_embed(pos))
        return self.proj(patch) + self.pos_embed(pos)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patch": self.num_patch,
            }
        )
        return config


ish = (10, 128, 128, 1)
xs = []
inp = Input(ish)
print(inp)


# @keras.saving.register_keras_serializable(package="MyLayers")
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patch": self.num_patch, "embed_dim": self.embed_dim})
        return config


for slice_indx in range(0, 10, 1):
    x = Lambda(lambda x: x[:, slice_indx])(inp)
    x = layers.RandomCrop(image_dimension, image_dimension)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation="relu")(x)

    xs.append(x)

t = t = Lambda(list_to_stack)(xs)
t = Flatten()(t)
t = layers.Dense(7, activation="softmax")(t)

# t=Lambda(list_to_stack)(xs)
# t=Conv3D(5,3,padding='same')(t)
# t=BatchNormalization(momentum=0.8)(t)
# target_shape=(7,32*48*50)
# t=Reshape(target_shape)(t)
# t=GRU(25, return_sequences=True)(t)
# t=GRU(50, return_sequences=False,dropout=0.5)(t)

# t=Dense(100,'relu')(t)
# t = Flatten()(t)
# out=Dense(7, activation='softmax')(t)

# model = Model(inputs=inp, outputs=out)
# opt = tf.keras.optimizers.SGD(lr=0.0087)
# model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
# model.summary()

model = keras.Model(inputs=inp, outputs=t)
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
model.summary()

# assign location
print(os.getcwd())
path='./src/Vision/saves/actions'

model.load_weights(path)

# Load and process the image
# img_path = './src/Vision/read10_rgb_2_frame31.png'

# img = load_img(img_path, target_size=(128, 128))  # assuming you need the image to be 128x128
# img = img_to_array(img)
# img = np.expand_dims(img, axis=0)  # model.predict expects a batch of images

# testarr= np.load('./src/Vision/testarr.npy')

# predicted  = model.predict(testarr,batch_size = 10)
# predicted  = np.argmax(predicted,axis=1)
# print(predicted)
# labels = ['Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping', 'Standing', 'Turning_Around', 'Writting']
# for i in predicted :
#     print(labels[i])



def predict_action (batch):
    predicted  = model.predict(batch,batch_size = 10)
    predicted  = np.argmax(predicted,axis=1)
    labels = ['Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping', 'Standing', 'Turning_Around', 'Writting']
    predictions=[]
    for i in predicted :
        predictions.append(labels[i])
    return predictions