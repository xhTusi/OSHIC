# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:18:35 2020

@author: sj
"""

from __future__ import print_function
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import concatenate, Dense, Add, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Reshape
import math


class Convdown(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Convdown, self).__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim * 2, 1, strides=1, padding='valid'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(dim * 2, 3, strides=1, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(dim, 1, strides=1, padding='valid')
        ])

        self.attn = ESSAttn(dim)
        self.norm = keras.layers.BatchNormalization(axis=-1)
        self.drop = keras.layers.Dropout(0.2)

    def call(self, x):
        shortcut = x
        x_size = (x.shape[1], x.shape[2])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))  # + x_embed
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = tf.concat([x, shortcut], axis=-1)
        x = self.convd(x)
        x = x + shortcut
        return x

class Convup(tf.keras.Model):
    def __init__(self, dim):
        super(Convup, self).__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convu = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim * 2, kernel_size=1, strides=1, padding='valid'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(dim * 2, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, padding='valid')
        ])
        self.drop = tf.keras.layers.Dropout(0.2)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = ESSAttn(dim)

    def call(self, x):
        shortcut = x
        x_size = (x.shape[1], x.shape[2])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = tf.concat((x, shortcut), axis=-1)
        x = self.convu(x)
        x = x + shortcut
        return x

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self):
        super(PatchEmbed, self).__init__()

    def call(self, x):
        _, H, W, C = x.shape
        B= tf.shape(x)[0]
        x = tf.reshape(x, (B, H * W, C))
        #x = tf.transpose(x, perm=[0, 2, 1])
        return x

class PatchUnEmbed(tf.keras.layers.Layer):
    def __init__(self, in_chans=3, embed_dim=96):
        super(PatchUnEmbed, self).__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def call(self, x, x_size):
        _, HW, C = x.shape
        B = tf.shape(x)[0]
        #x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, (B, x_size[0], x_size[1], self.embed_dim))
        return x

class ESSAttn(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(ESSAttn, self).__init__(**kwargs)
        self.dim = dim
        self.lnqkv = tf.keras.layers.Dense(dim * 3)
        self.ln = tf.keras.layers.Dense(dim)

    def call(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        q, k, v = tf.split(qkv, 3, axis=2)
        a = tf.reduce_mean(q, axis=2, keepdims=True)
        q = q - a
        a = tf.reduce_mean(k, axis=2, keepdims=True)
        k = k - a
        q2 = tf.pow(q, 2)
        q2s = tf.reduce_sum(q2, axis=2, keepdims=True)
        k2 = tf.pow(k, 2)
        k2s = tf.reduce_sum(k2, axis=2, keepdims=True)
        t1 = v
        k2 = tf.nn.l2_normalize(k2 / (k2s + 1e-7), axis=-2)
        q2 = tf.nn.l2_normalize(q2 / (q2s + 1e-7), axis=-1)
        t2 = q2 @ tf.linalg.matmul(k2, v, transpose_a=True, transpose_b=False) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        return attn

class Blockup(tf.keras.Model):
    # def __init__(self, dim, upscale):
    def __init__(self, dim):
        super(Blockup, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)

    def call(self, x):
        x1 = self.convup(x)   # ==
        x2 = self.convdown(x1) #==
        x3 = self.convdown(x2) + x1     #==
        x4 = self.convdown(x3) + x2     #==
        x5 = self.convdown(x4)

        return x5

def img2windows(img, H_sp, W_sp):
    _, C, H, W = img.shape
    B = tf.shape(img)[0]
    img_reshape = tf.reshape(img, [B, C, H // H_sp, H_sp, W // W_sp, W_sp])
    img_perm = tf.transpose(img_reshape, [0, 2, 4, 3, 5, 1])
    img_perm = tf.reshape(img_perm, [-1, H_sp * W_sp, C])
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):

    B = int(tf.cast(tf.shape(img_splits_hw)[0], tf.float32) / (H * W / H_sp / W_sp))

    img = tf.reshape(img_splits_hw, [B, H // H_sp, W // W_sp, H_sp, W_sp, -1])
    img = tf.transpose(img, [0, 1, 3, 2, 4, 5])
    img = tf.reshape(img, [B, H, W, -1])
    return img

class DynamicPosBias(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = tf.keras.layers.Dense(self.pos_dim * 2)
        self.pos1 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.pos_dim)
        ])
        self.pos2 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.pos_dim)
        ])
        self.pos3 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.num_heads)
        ])

    def call(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, dim, idx, split_size=[8, 8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            position_bias_h = tf.range(1 - self.H_sp, self.H_sp)
            position_bias_w = tf.range(1 - self.W_sp, self.W_sp)
            biases = tf.transpose(tf.meshgrid(position_bias_h, position_bias_w))
            biases = tf.reshape(biases, (2, -1))
            self.rpe_biases = tf.Variable(biases, trainable=False)

            coords_h = tf.range(self.H_sp)
            coords_w = tf.range(self.W_sp)
            coords = tf.transpose(tf.meshgrid(coords_h, coords_w))
            coords_flatten = tf.reshape(coords, (-1, 2))
            relative_coords = tf.expand_dims(coords_flatten, axis=1) - tf.expand_dims(coords_flatten, axis=0)
            relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
            relative_coords = relative_coords.numpy()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
            self.relative_position_index = tf.Variable(relative_position_index, trainable=False)

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)

    def im2win(self, x, H, W):
        _, N, C = x.shape
        B = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, (B, C, H, W))
        x = img2windows(x, self.H_sp, self.W_sp)
        x = tf.reshape(x, (-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, qkv, H, W, mask=None):
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))

        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = tf.gather(pos, tf.reshape(self.relative_position_index, [-1]))
            relative_position_bias = tf.reshape(relative_position_bias, [self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1])
            relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
            attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        N = attn.shape[3]

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, (B, nW, self.num_heads, N, N)) + tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (-1, self.H_sp * self.W_sp, C))

        x = windows2img(x, self.H_sp, self.W_sp, H, W)

        return x

class AdaptiveChannelAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(AdaptiveChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)

        self.dwconv = tf.keras.Sequential([
            keras.layers.Conv2D(dim, kernel_size=3, strides=1, padding='same', groups=dim, data_format='channels_first'),
            #keras.layers.LayerNormalization(epsilon=1e-5),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.Activation('gelu')
        ])

        self.channel_interaction = tf.keras.Sequential([
            keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1])),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Lambda(lambda x: tf.reshape(x,(-1, dim, 1, 1))),
            keras.layers.Conv2D(dim // 8, kernel_size=1, data_format='channels_first'),
            keras.layers.BatchNormalization(axis=1),
            #keras.layers.LayerNormalization(epsilon=1e-5),
            keras.layers.Activation('gelu'),
            keras.layers.Conv2D(dim, kernel_size=1, data_format='channels_first')
        ])

        self.spatial_interaction = tf.keras.Sequential([
            keras.layers.Conv2D(dim // 16, kernel_size=1, data_format='channels_first'),
            keras.layers.BatchNormalization(axis=1),
            #keras.layers.LayerNormalization(epsilon=1e-5),
            keras.layers.Activation('gelu'),
            keras.layers.Conv2D(1, kernel_size=1, data_format='channels_first')
        ])

        self.temperature = tf.Variable(tf.ones((num_heads, 1, 1)), trainable=True)

    def call(self, x):
        _, C, H, W = x.shape
        B = tf.shape(x)[0]

        x = tf.reshape(x, (B, C, H * W))
        x = tf.transpose(x, (0, 2, 1))
        _, N, C = x.shape

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, self.dim // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = tf.transpose(q, (0, 1, 3, 2))
        k = tf.transpose(k, (0, 1, 3, 2))
        v = tf.transpose(v, (0, 1, 3, 2))
        v_ = tf.reshape(v, (B, C, N))
        v_ = tf.reshape(v_, (B, C, H, W))

        q = tf.nn.l2_normalize(q, axis=-1)
        k = tf.nn.l2_normalize(k, axis=-1)

        attn = (tf.matmul(q, k, transpose_b=True)) * self.temperature
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        attened_x = tf.matmul(attn, v)
        attened_x = tf.transpose(attened_x, (0, 3, 1, 2))
        attened_x = tf.reshape(attened_x, (B, N, C))

        conv_x = self.dwconv(v_)

        attention_reshape = tf.transpose(attened_x, (0, 2, 1))
        attention_reshape = tf.reshape(attention_reshape, (B, C, H, W))

        channel_map = self.channel_interaction(attention_reshape)
        spatial_map = self.spatial_interaction(conv_x)
        spatial_map = tf.transpose(spatial_map, (0, 2, 3, 1))
        spatial_map = tf.reshape(spatial_map, (B, N, 1))

        # S-I
        attened_x = attened_x * tf.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * tf.sigmoid(channel_map)
        conv_x = tf.transpose(conv_x, (0, 2, 3, 1))
        conv_x = tf.reshape(conv_x, (B, N, C))

        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = tf.reshape(x, (B, C, H, W))
        return x

class AdaptiveSpatialAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads, reso=64, split_size=[8, 8], shift_size=[1, 2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super(AdaptiveSpatialAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must be in the range [0, split_size[0])"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must be in the range [0, split_size[1])"

        self.branch_num = 2

        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(drop)

        self.attns = [
            SpatialAttention(
                dim // 2, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)
        ]

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
                self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.attn_mask_0 = tf.convert_to_tensor(attn_mask[0])
            self.attn_mask_1 = tf.convert_to_tensor(attn_mask[1])
        else:
            attn_mask = None
            self.attn_mask_0 = None
            self.attn_mask_1 = None

        self.dwconv = keras.Sequential([
            keras.layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", data_format='channels_first', groups=dim),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ReLU()
        ])#3->1

        self.channel_interaction = keras.Sequential([
            keras.layers.Lambda(lambda  x: tf.transpose(x, [0, 2, 3, 1])),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Lambda(lambda x:tf.reshape(x,(-1, dim, 1, 1))),
            keras.layers.Conv2D(dim // 8, kernel_size=1, data_format='channels_first'),
            keras.layers.LayerNormalization(epsilon=1e-5),
            #keras.layers.BatchNormalization(axis=1),
            keras.layers.ReLU(),
            keras.layers.Conv2D(dim, kernel_size=1, data_format='channels_first')
        ])

        self.spatial_interaction = keras.Sequential([
            keras.layers.Conv2D(dim // 16, kernel_size=1 ,data_format='channels_first',),
            keras.layers.LayerNormalization(epsilon=1e-5),
            #keras.layers.BatchNormalization(axis=1),
            keras.layers.ReLU(),
            keras.layers.Conv2D(1, kernel_size=1 ,data_format='channels_first',)
        ])

    def calculate_mask(self, H, W):
        img_mask_0 = tf.zeros((1, H, W, 1), dtype=tf.float32)
        img_mask_1 = tf.zeros((1, H, W, 1), dtype=tf.float32)
        h_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        img_mask_0 = tf.reshape(img_mask_0, (1, H // self.split_size[0], self.split_size[0], W // self.split_size[1],
                                             self.split_size[1], 1))
        img_mask_0 = tf.transpose(img_mask_0, perm=(0, 1, 3, 2, 4, 5))
        img_mask_0 = tf.reshape(img_mask_0, (-1, self.split_size[0], self.split_size[1], 1))
        mask_windows_0 = tf.reshape(img_mask_0, (-1, self.split_size[0] * self.split_size[1]))
        attn_mask_0 = mask_windows_0[:, tf.newaxis] - mask_windows_0[:, tf.newaxis, :]
        attn_mask_0 = tf.where(attn_mask_0 != 0, -100.0, 0.0)

        img_mask_1 = tf.reshape(img_mask_1, (1, H // self.split_size[1], self.split_size[1], W // self.split_size[0],
                                             self.split_size[0], 1))
        img_mask_1 = tf.transpose(img_mask_1, perm=(0, 1, 3, 2, 4, 5))
        img_mask_1 = tf.reshape(img_mask_1, (-1, self.split_size[1], self.split_size[0], 1))
        mask_windows_1 = tf.reshape(img_mask_1, (-1, self.split_size[1] * self.split_size[0]))
        attn_mask_1 = mask_windows_1[:, tf.newaxis] - mask_windows_1[:, tf.newaxis, :]
        attn_mask_1 = tf.where(attn_mask_1 != 0, -100.0, 0.0)

        return attn_mask_0, attn_mask_1

    def call(self, x):
        _, C, H, W = x.shape
        x = tf.transpose(x, (0, 2, 3, 1))
        x = tf.reshape(x, [-1, H * W, self.dim])

        B, L, C = x.shape
        B = tf.shape(x)[0]
        assert L == H * W, "Flattened img_tokens have the wrong size"

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, L, 3, C])
        qkv = tf.transpose(qkv, perm=(2, 0, 1, 3))  # 3, B, HW, C
        v = qkv[2]
        v = tf.transpose(v, perm=(0, 2, 1))
        v = tf.reshape(v, (B, C, H, W))

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size


        qkv = tf.reshape(qkv, [3 * B, H, W, C])
        qkv = tf.transpose(qkv, (0, 3, 1, 2))
        qkv = tf.pad(qkv, [[0, 0], [0, 0], [pad_l, pad_r], [pad_t, pad_b]])
        qkv = tf.reshape(qkv, [3, B, C, (H + pad_t + pad_b) * (W + pad_l + pad_r)])
        qkv = tf.transpose(qkv, perm=(0, 1, 3, 2))

        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
                self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
            qkv = tf.reshape(qkv, [3, B, _H, _W, C])
            qkv_0 = tf.roll(qkv[:, :, :, :, :C // 2], shift=(-self.shift_size[0], -self.shift_size[1]), axis=(2, 3))
            qkv_0 = tf.reshape(qkv_0, [3, B, _L, C // 2])
            qkv_1 = tf.roll(qkv[:, :, :, :, C // 2:], shift=(-self.shift_size[1], -self.shift_size[0]), axis=(2, 3))
            qkv_1 = tf.reshape(qkv_1, [3, B, _L, C // 2])

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0])
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1])
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = tf.roll(x1_shift, shift=(self.shift_size[0], self.shift_size[1]), axis=(1, 2))
            x2 = tf.roll(x2_shift, shift=(self.shift_size[1], self.shift_size[0]), axis=(1, 2))
            x1 = x1[:, :H, :W, :]
            x2 = x2[:, :H, :W, :]
            attened_x = tf.concat([x1, x2], axis=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], _H, _W)[:, :H, :W, :]
            x1 = tf.reshape(x1, [B, L, C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :]
            x2 = tf.reshape(x2, [B, L, C // 2])
            attened_x = tf.concat([x1, x2], axis=2)

        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)

        # C - Map
        channel_map = self.channel_interaction(conv_x)
        channel_map = tf.transpose(channel_map, perm=(0, 2, 3, 1))
        channel_map = tf.reshape(channel_map, [B, 1, C])

        # S - Map
        attention_reshape = tf.transpose(attened_x, perm=(0, 2, 1))
        attention_reshape = tf.reshape(attention_reshape, [B, C, H, W])
        spatial_map = self.spatial_interaction(attention_reshape)
        spatial_map = tf.transpose(spatial_map, perm=(0, 1, 2, 3))

        # C - I
        attened_x = attened_x * tf.sigmoid(channel_map)

        # S - I
        conv_x = tf.sigmoid(spatial_map) * conv_x
        conv_x = tf.transpose(conv_x, perm=(0, 2, 3, 1))
        conv_x = tf.reshape(conv_x, [B, L, C])

        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = tf.reshape(x, [-1, C, H, W])

        return x

import tensorflow as tf
from tensorflow.keras import layers, models

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, shift_size=0, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = models.Sequential([
            layers.Dense(dim * 4),
            layers.Activation(tf.nn.gelu),
            layers.Dense(dim)
        ])

    def call(self, x):
        # x: [batch_size, height, width, channels]
        _, H, W, C = x.shape
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, H * W, C))  # B, N, C

        # Swin Transformer attention mechanism
        # (Omitting the window partitioning and shifting for simplicity)

        x = self.norm1(x)
        x = self.attn(x, x)  # The output of attn automatically includes the residual connection
        x = self.norm2(x)

        x = x + self.mlp(x)

        x = tf.reshape(x, (B, H, W, C))

        return x


class SwinTransformer(tf.keras.Model):
    def __init__(self, dim, depths, num_heads):
        super(SwinTransformer, self).__init__()
        self.swin_layers = []
        for depth, head in zip(depths, num_heads):
            layer = [SwinTransformerBlock(dim, head) for _ in range(depth)]
            self.swin_layers.append(layer)

    def call(self, x):
        for layer in self.swin_layers:
            for block in layer:
                x = block(x)
        return x

def resnet99_avg_recon(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    Essa = Blockup(dim = 64)

    AsSa_n1 = AdaptiveSpatialAttention(
        dim=32, num_heads=8, reso=64, split_size=[2, 4], shift_size=[1, 2], qkv_bias=False, qk_scale=None,
        drop=0.,
        attn_drop=0., rg_idx=0, b_idx=0
    )
    AcSa_n1 = AdaptiveChannelAttention(
        dim=32, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0
    )
    AsSa_n2 = AdaptiveSpatialAttention(
        dim=64, num_heads=8, reso=64, split_size=[2, 4], shift_size=[1, 2], qkv_bias=False, qk_scale=None,
        drop=0.,
        attn_drop=0., rg_idx=0, b_idx=0
    )
    AcSa_n2 = AdaptiveChannelAttention(
        dim=64, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0
    )
    Sa = SwinTransformer(dim=64, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],)

    #
    dconv1 = Conv2DTranspose(64, kernel_size=(1,1), padding='valid')
    dconv2 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv3 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv4 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv5 = Conv2DTranspose(band, kernel_size=(3,3), padding='valid')
    bn1_de = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')
    bn2_de = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')

    if l==1:# mdl4ow
        x1 = conv0(input1)
        x1x = conv0x(input1)
        #    x1 = MaxPooling2D(pool_size=(2,2))(x1)
        #    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
        x1 = concatenate([x1, x1x], axis=-1)
        x11 = bn11(x1)
        x11 = Activation('relu')(x11)
        x11 = conv11(x11)
        x11 = Activation('relu')(x11)
        x11 = conv12(x11)
        x1 = Add()([x1, x11])


    if l==2:#dual
        x1 = conv0(input1)
        x1x = conv0x(input1)

        x1 = tf.transpose(x1, perm=[0, 3, 1, 2])  #BHWC -> BCHW
        x1 = AsSa_n1(x1)        #64-dim
        x1 = AcSa_n1(x1)        #64-dim
        x1 = tf.transpose(x1, perm=[0, 2, 3, 1])

        x1 = concatenate([x1,x1x],axis=-1)
        x11 = bn11(x1)
        x11 = Activation('relu')(x11)
        #
        x1 = tf.transpose(x1, perm=[0, 3, 1, 2])  #BHWC -> BCHW
        x1 = AsSa_n2(x1)        #64-dim
        x1 = AcSa_n2(x1)        #64-dim
        x1 = tf.transpose(x1, perm=[0, 2, 3, 1])

        x11 = Activation('relu')(x11)
        x11 = conv12(x11)
        x1 = Add()([x1,x11])


    if l==3:#essa
        x1 = conv0(input1)
        x1x = conv0x(input1)

        x1 = concatenate([x1,x1x],axis=-1)
        x11 = bn11(x1)
        x11 = Activation('relu')(x11)
        x11 = Essa(x11)
        x11 = Activation('relu')(x11)
        x11 = conv12(x11)
        x1 = Add()([x1,x11])

    if l == 4:
        x1 = conv0(input1)
        x1x = conv0x(input1)

        x1 = concatenate([x1, x1x], axis=-1)
        x11 = bn11(x1)
        x11 = Activation('relu')(x11)
        x11 = Sa(x11)
        x11 = Activation('relu')(x11)
        x11 = conv12(x11)
        x1 = Add()([x1, x11])


    x1 = GlobalAveragePooling2D(name='ploss')(x1)
    pre1 = fc1(x1)
    
    #
    x12 = Reshape((1,1,64))(x1)
    x12 = dconv1(x12)
    x12 = bn1_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv2(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv3(x12)
    x12 = bn2_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv4(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv5(x12)

    model1 = Model(inputs=input1, outputs=[pre1,x12])
    model2 = Model(inputs=input1, outputs=pre1)
    return model1,model2


def wcrn_recon(band, ncla1):
    input1 = Input(shape=(5, 5, band))

    # define network
    conv0x = Conv2D(64, kernel_size=(1, 1), padding='valid',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(64, kernel_size=(3, 3), padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros',
                              moving_variance_initializer='ones')
    conv11 = Conv2D(128, kernel_size=(1, 1), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(128, kernel_size=(1, 1), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    #
    fc1 = Dense(ncla1, activation='softmax', name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    #
    bn_de1 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')
    dconv1 = Conv2DTranspose(128, kernel_size=(1, 1), padding='valid')
    dconv2 = Conv2DTranspose(128, kernel_size=(1, 1), padding='valid')
    dconv3 = Conv2DTranspose(128, kernel_size=(3, 3), padding='valid')
    dconv4 = Conv2DTranspose(band, kernel_size=(3, 3), padding='valid')

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
    x1 = MaxPooling2D(pool_size=(3, 3))(x1)
    x1x = MaxPooling2D(pool_size=(5, 5))(x1x)
    x1 = concatenate([x1, x1x], axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add(name='ploss')([x1, x11])

    x11 = Flatten()(x1)
    pre1 = fc1(x11)

    #    x12 = dconv1(x1)
    #    x12 = Activation('relu')(x12)
    #    x12 = dconv2(x12)
    #    x12 = Activation('relu')(x12)
    #    x12 = dconv3(x12)

    x12 = bn_de1(x1)
    x12 = Activation('relu')(x12)
    x12 = dconv1(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv2(x12)
    x12 = Add()([x1, x12])
    x12 = dconv3(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv4(x12)
    print("wcrn is !!!!!!")

    model1 = Model(inputs=input1, outputs=[pre1, x12])
    model2 = Model(inputs=input1, outputs=pre1)
    return model1, model2