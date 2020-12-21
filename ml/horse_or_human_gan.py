import logging

import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Activation
from keras.models import Model
from scipy.spatial.distance import euclidean
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from .logger import get_logger
from .utils import *


class HorseOrHumanGAN:
    def __init__(self, batch_size=32, filters=(32, 64), latent_dim=16,
                 dataset_dir="horses_or_humans",
                 log_file="logs.log", log_level=logging.INFO):
        """Class with AutoEncoder model to classify Horses and Humans

        :param batch_size: Batch size
        :type batch_size: int
        :param filters: Filter levels
        :type filters: list or tuple
        :param latent_dim: Number of layers in the middle of autoencoder
        :type latent_dim: int
        :param dataset_dir: Path to dataset directory
        :type dataset_dir: str
        :param log_file: FilePath to save logs
        :type log_file: str
        :param log_level: Logger level from "logging" library
        :type log_level: logging.BASIC_FORMAT
        """

        self.logger = get_logger(log_file, log_level, __name__)

        # Initialize class variables
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.input_shape = (300, 300, 3)
        self.volume_size = None

        self.train_url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
        self.validate_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

        # Init models
        self.logger.info("Initializing Models")
        inputs = Input(shape=self.input_shape)
        self.encoder = self._build_encoder(filters, latent_dim, inputs)
        self.decoder = self._build_decoder(filters, latent_dim)

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name="autoencoder")
        self.optimizer = Adam()
        self.autoencoder.compile(optimizer=self.optimizer, loss="mse")

        self.logger.info("Horse Or Human GAN class initialized")

    def _get_generator(self, dataset_dir, batch_size, data_type=0,
                       dataset_url=None, temp_zip_path="horses_or_humans.zip"):
        """Function to download dataset, if needed, and create generator

        :param data_type: Type of data (0 - labeled as in dir; 1 - input=output, for autoencoders)
        :type data_type: int
        :param dataset_dir: Path to dataset dir
        :type dataset_dir: str
        :param dataset_url: Url path to dataset
        :type dataset_url: str
        :param temp_zip_path: Path to temporary save zip file of dataset, if needed
        :type temp_zip_path: str
        """
        self.logger.info(f"Initializing Dataset from {dataset_dir}")
        assert data_type in (0, 1), f"data_type is {data_type}, but can only be 0 or 1"

        if dataset_url is None:
            self.logger.info("Downloading dataset from server")
            dataset_url = self.train_url

        if os.path.exists(self.dataset_dir):
            pass  # TODO: Check dataset path
        else:
            if dataset_url:
                download_file(dataset_url, temp_zip_path)
                if dataset_url.endswith("zip"):
                    unzip_archive(temp_zip_path, self.dataset_dir)
            else:
                raise ValueError(f"Dataset not found in {dataset_dir} dir, and dataset_url is None")

        train_datagen = ImageDataGenerator(rescale=1. / 255)

        if data_type == 0:
            return train_datagen.flow_from_directory(
                self.dataset_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode="binary",
                shuffle=True)
        elif data_type == 1:
            return train_datagen.flow_from_directory(
                self.dataset_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
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

    def train_autoencoder(self, epochs):
        """Function to train autoencoder

        :param epochs: Number of epochs
        :type epochs: int
        """
        self.logger.info("Training autoencoder")
        callbacks = [EarlyStopping(monitor='loss', patience=10),
                     ModelCheckpoint('autoencoder_weights.hdf5', monitor='loss', save_best_only=True, verbose=1)]

        generator = self._get_generator(self.dataset_dir, self.batch_size, data_type=1)
        self.autoencoder.fit(generator, epochs=epochs, callbacks=callbacks)  # TODO: Add callback to save logs

    def load_weights(self, autoencoder_path, autoencoder_url=None):  # TODO: Comments
        if not os.path.exists(autoencoder_path) and autoencoder_url:
            self.logger.info(f"Downloading weights from {autoencoder_url}")
            download_file(autoencoder_url, autoencoder_path)
            if autoencoder_url.endswith("zip"):
                unzip_archive(autoencoder_path, autoencoder_path)

        self.logger.info(f"Loading weights from {autoencoder_path}")
        self.autoencoder.load_weights(autoencoder_path)  # TODO: Load weights to server

    @staticmethod
    def _image_preprocessing(images):
        """Pre processing image before predict"""
        images = np.array(images)
        if len(images.shape) == 3:
            images = np.array([images, ])

        images = images.astype(np.float64) / 255

        return images

    @staticmethod
    def _image_postprocessing(prediction):
        """Post processing image after predict"""
        result = []
        for image in prediction:
            image = np.array(image) * 255
            image = image.astype(np.uint8)
            result.append(image)
        return result

    def predict_encoder(self, images):
        """Function to predict results from encoder

        :param images: Image array
        :type images: np.ndarray or list[np.ndarray]
        """

        images = self._image_preprocessing(images)
        prediction = self.encoder(images)
        return prediction

    def count_distances(self, test_dataset_dir):
        distances = {
            "Horse_Horse": [],
            "Human_Human": [],
            "Horse_Human": []
        }

        generator = self._get_generator(test_dataset_dir, batch_size=2, dataset_url=self.validate_url)

        for i, d in enumerate(generator):
            if i == 10000:
                break

            data, labels = d
            if len(data) == 2:
                prediction = self.predict_encoder(data)
                distance = euclidean(prediction[0], prediction[1])

                if all(labels == [0, 0]):
                    distances["Horse_Horse"].append(distance)
                elif all(labels == [1, 1]):
                    distances["Human_Human"].append(distance)
                else:
                    distances["Horse_Human"].append(distance)

        for key in distances:
            distances[key] = sum(distances[key]) / len(distances[key])

        return distances
