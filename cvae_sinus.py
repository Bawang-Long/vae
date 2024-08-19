#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:22:30 2024

@author: marlon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
#import tensorflow_probability as tfp
import time
from sklearn.model_selection import train_test_split

import keras
from keras import ops
from keras import layers

#data prepare
# 设置随机种子以便复现结果
np.random.seed(0)

# 参数定义
num_samples = 10000
num_timesteps = 1000
num_features = 1

# 创建正弦信号 X
# 生成时间序列
t = np.linspace(0, 2 * np.pi, num_timesteps)

# 初始化信号数组
X = np.zeros((num_samples, num_timesteps, num_features))

# 生成随机正弦信号
for i in range(num_samples):
    # 生成随机周期、幅值和相位
    frequency = np.random.uniform(0.1, 10)   # 周期范围 [0.1, 10]
    amplitude = np.random.uniform(0.5, 2)    # 幅值范围 [0.5, 2]
    phase = np.random.uniform(0, 2 * np.pi)  # 相位范围 [0, 2π]

    # 生成正弦信号
    signal = amplitude * np.sin(frequency * t + phase)

    # 将信号加入数组，并增加一个维度以匹配 (num_samples, num_timesteps, num_features)
    X[i, :, 0] = signal


# 创建随机噪声信号 Y
Y = np.random.normal(0, 1, (num_samples, num_timesteps, num_features))

# 创建叠加信号 Z
Z = X + Y

# split dataset
X_train, X_test, Y_train, Y_test, Z_train, Z_test,  = train_test_split(X, Y, Z, test_size=0.2, random_state=42)


#%%
#Create a sampling layer
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.seed_generator = keras.random.SeedGenerator(1337)

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = ops.shape(z_mean)[0]
#         dim = ops.shape(z_mean)[1]
#         epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
#         return z_mean + ops.exp(0.5 * z_log_var) * epsilon
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



#Build the encoder
#Parameter
latent_dim = 16

encoder_inputs = keras.Input(shape=(num_timesteps, num_features))

x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)

x = layers.Flatten()(x)
#x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()




#Build the decoder for X and Y
latent_inputs = keras.Input(shape=(latent_dim,))

x = layers.Dense(16000, activation="relu")(latent_inputs)

x = layers.Reshape((250, 64))(x)

x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

#x = layers.Conv1DTranspose(1, 3, activation="relu", padding="same")(x)

x = layers.Flatten()(x)

X = layers.Dense(num_timesteps)(x)
X_outputs = layers.Reshape(target_shape = (1000 ,1), name="X_outputs")(X)
Y = layers.Dense(num_timesteps)(x)
Y_outputs = layers.Reshape(target_shape = (1000 ,1), name="Y_outputs")(Y)
decoder = keras.Model(latent_inputs, [X_outputs , Y_outputs], name="decoder")
decoder.summary()

#Build the decoder for Y
# latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(16000, activation="relu")(latent_inputs)
# x = layers.Reshape((250, 64))(x)
# x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
# decoderY = keras.Model(latent_inputs, decoder_outputs, name="decoderY")
# decoderY.summary()


#Define the VAE as a Model with a custom train_step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        X_train = data[..., 0:1]
        Y_train = data[..., 1:2]
        Z_train = data[..., 2:3]
    # def train_step(self, Z_train, XY_train):
    #     X_train = XY_train[..., 0:1]
    #     Y_train = XY_train[..., 1:2]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(Z_train)
            X_rec, Y_rec = self.decoder(z)
      
            X_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(X_train, X_rec),axis=1))
            Y_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(Y_train, Y_rec),axis=1))
            #Y_loss = tf.reduce_mean(keras.losses.mean_squared_error(Y_train, Y_rec))
            # X_loss = keras.losses.mean_squared_error(X_train, X_rec)
            # Y_loss = keras.losses.mean_squared_error(Y_train, Y_rec)
            reconstruction_loss = X_loss + Y_loss
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#XY_train =np.concatenate(( X_train, Y_train), axis = 2)
data =np.concatenate(( X_train, Y_train, Z_train), axis = 2)
# a = XY_train[..., 0:1]
# b = XY_train[..., 1:2]
#XY_train = (X_train, Y_train)
vae =  VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(data, epochs=10, batch_size= 16)

#%%

def generate_and_visualize(model, test_sample):
    # 获取模型的输出
    z_mean, z_log_var, z = model.encoder.predict(test_sample)
    X_rec, Y_rec = model.decoder.predict(z)
    #print(z_mean,z_log_var)
    #print(z)
    num_samples = 16
    print(X_rec)

    fig_x, axes_x = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_samples):
        X_sample = X_rec[i]
        ax = axes_x[i // 4, i % 4]
        ax.plot(X_sample, color='blue')
        ax.set_title(f'X Sample {i+1}')


    plt.tight_layout()
    plt.suptitle("X Outputs", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


    fig_y, axes_y = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_samples):
        Y_sample = Y_rec[i]
        ax = axes_y[i // 4, i % 4]
        ax.plot(Y_sample, color='orange')
        ax.set_title(f'Y Sample {i+1}')

    plt.tight_layout()
    plt.suptitle("Y Outputs", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


#visualize test dataset
X_true = X_test[:16]
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.plot(X_true[i].squeeze())
    ax.set_title(f"Sample {i+1}")
plt.suptitle("X_true", fontsize=16)
plt.tight_layout()
plt.show()

Y_true = Y_test[:16]
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.plot(Y_true[i].squeeze())
    ax.set_title(f"Sample {i+1}")
plt.suptitle("Y_true", fontsize=16)
plt.tight_layout()
plt.show()

test_sample = Z_test[:16]
#print(test_sample, test_sample.shape)
generate_and_visualize(vae, test_sample)
#generate(vae, test_sample)











