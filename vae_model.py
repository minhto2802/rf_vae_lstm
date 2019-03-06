import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.vis_utils import plot_model
import numpy as np
from keras import objectives


def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.,
                    act_type=None):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    """Source code: https://github.com/twairball/keras_lstm_vae"""

    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim, activation=act_type)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim, activation=act_type)(h)
    z_log_var = Dense(latent_dim, activation=act_type)(h)

    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
            # Arguments
                args (tensor): mean and log of variance of Q(z|X)
            # Returns
                z (tensor): sampled latent vector
            """
        _z_mean, _z_log_var = args
        batch_sz = K.shape(_z_mean)[0]
        dim = K.int_shape(_z_mean)[1]
        epsilon = K.random_normal(shape=(batch_sz, dim),
                                  mean=0., stddev=epsilon_std)
        # return _z_mean + K.exp(0.5 * _z_log_var) * epsilon
        return _z_mean + _z_log_var * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # decoding LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True, activation=act_type)
    decoder_mean = LSTM(input_dim, return_sequences=True, activation=act_type)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoding layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean, use_mse=False):
        if use_mse:
            reconstruction_loss = objectives.mse(x, x_decoded_mean)
        else:
            reconstruction_loss = objectives.binary_crossentropy(x, x_decoded_mean)

        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        # kl_loss = K.sum(kl_loss, axis=-1)
        # kl_loss *= -0.5
        loss = reconstruction_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator
