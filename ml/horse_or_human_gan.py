import logging

import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from .logger import get_logger
from .utils import *


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
        self.batch_size = batch_size
        self.input_shape = (300, 300 ,3)
        self.volume_size = None

        # Download and prepare dataset
        self.logger.info("Initializing Dataset")
        self.train_generator = self._get_generator(data_type=1)

        # Init models
        self.logger.info("Initializing Models")
        inputs = Input(shape=self.input_shape)
        self.encoder = self._build_encoder(filters, latent_dim, inputs)
        self.decoder = self._build_decoder(filters, latent_dim)

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name="autoencoder")
        self.optimizer = Adam()
        self.autoencoder.compile(optimizer=self.optimizer, loss="mse")

        self.logger.info("Horse Or Human GAN class initialized")

    def _get_generator(self, data_type=0, dataset_dir="horses_or_humans", dataset_url=None,
                       temp_zip_path="horses_or_humans.zip"):
        """Function to dowmload dataset, if needed, and create generator

        :param data_type: Type of data (0 - labeled as in dir, 1 - input=output for autoencoders)
        :type data_type: int
        :param dataset_dir: Path to dataset directory
        :type dataset_dir: str
        :param dataset_url: Url path to dataset
        :type dataset_url: str
        :param temp_zip_path: Path to temporary save zip file of dataset, if needed
        :type temp_zip_path: str
        """
        assert data_type in (1, 2), f"data_type is {data_type}, but can be only 0 or 1"

        if dataset_url is None:
            dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"

        if os.path.exists(dataset_dir):
            pass  # TODO: Check dataset path
        else:
            download_file(dataset_url, temp_zip_path)
            if dataset_url.endswith("zip"):
                unzip_archive(temp_zip_path, dataset_dir)

        train_datagen = ImageDataGenerator(rescale=1. / 255)

        if data_type == 0:
            return train_datagen.flow_from_directory(
                dataset_dir,
                target_size=self.input_shape[:2],
                batch_size=self.batch_size,
                shuffle=True)
        elif data_type == 1:
            return train_datagen.flow_from_directory(
                dataset_dir,
                target_size=self.input_shape[:2],
                batch_size=self.batch_size,
                class_mode="input",
                shuffle=True)

    def _build_encoder(self, filters, latent_dim, inputs):
        """Function to build encoder

        :param filters: Filter levels
        :type filters: list or tuple
        :param latent_dim: Number of layers in the middle of autoencoder
        :type latent_dim: int
        :param inputs: Input layer
        :type inputs: keras.layers.Input
        """
        self.logger.debug("Initializing encoder")

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
        self.logger.debug("Initializing decoder")

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
