import logging

import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Activation
from keras.models import Model
from keras.optimizers import Adam
from tensorflow_datasets import load

from .logger import get_logger

class HorseOrHumanGAN:
    def __init__(self, batch_size=32, filters=(32, 64), latent_dim=16,
                 log_file="logs.log", log_level=logging.INFO):
        """Class with AutoEncoder model to classify Horses and Humans

        :param batch_size: Batch size
        :type batch_size: int
        :param filters: Filter levels
        :type filters: list or tuple
        :param latent_dim: Number of layers in the middle of autoencoder
        :type latent_dim: int
        :param log_file: FilePath to save logs
        :type log_file: str
        :param log_level: Logger level from "logging" library
        :type log_level: logging.BASIC_FORMAT
        """

        self.logger = get_logger(log_file, log_level, __name__)

        # Initialize class variables
        self.input_shape = input_shape
        self.volume_size = None

        # Prepare dataset
        self.logger.info("Initializing Dataset")

        # Init models
        self.logger.info("Initializing Models")
        inputs = Input(shape=self.input_shape)
        self.encoder = self._build_encoder(filters, latent_dim, inputs)
        self.decoder = self._build_decoder(filters, latent_dim)

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name="autoencoder")
        self.optimizer = Adam()
        self.autoencoder.compile(optimizer=self.optimizer, loss="mse")

        self.logger.info("Horse Or Human GAN class initialized")
    def _build_encoder(self, filters, latent_dim, inputs):
        """Function to build encoder

        :param filters: Filter levels
        :type filters: list or tuple
        :param latent_dim: Number of layers in the middle of autoencoder
        :type latent_dim: int
        :param inputs: Input layer
        :type inputs: keras.layers.Input
        """
        # Define the input to the encoder
        x = inputs

        # Apply a CONV => RELU => BN operation
        for f in filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=-1)(x)

        # Flatten the network and then construct our latent vector
        self.volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latent_dim)(x)

        # Build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        return encoder

    def _build_decoder(self, filters, latent_dim):
        """Function to build decoder

        :param filters: Filter levels
        :type filters: list or tuple
        :param latent_dim: Number of layers in the middle of autoencoder
        :type latent_dim: int
        """

        # Accept on input encoder output
        latent_input = Input(shape=(latent_dim,))

        # Reshape flatten Dense
        x = Dense(np.prod(self.volume_size[1:]))(latent_input)
        x = Reshape((self.volume_size[1], self.volume_size[2], self.volume_size[3]))(x)

        # Apply a CONV_TRANSPOSE => RELU => BN operation
        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3, 3), strides=2,
                                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=-1)(x)

        # Recover image original depth
        x = Conv2DTranspose(self.input_shape[2], (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)

        # Build the decoder model
        decoder = Model(latent_input, outputs, name="decoder")

        return decoder