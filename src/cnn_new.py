import numpy as np
import pickle
import datetime as dt
from typing import Tuple, Union, List

import sklearn.metrics.pairwise
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.constraints import max_norm

from sklearn.metrics.pairwise import cosine_distances


CALLABLE = {
    TensorBoard: {
        'log_dir': '../logs/',
        'update_freq': 'epoch'
    },
    EarlyStopping: {}
}


class BaseAutoencoder:
    def __init__(self,
                 gray_images: bool = True,
                 use_gpu: bool = False):
        self.gray_images = gray_images
        self.encoder = None
        self.latent = None
        self.autoencoder = None
        self.config = None

        if use_gpu:
            self.config = tf.compat.v1.ConfigProto()
            self.config.gpu_options.allow_growth = True
            tf.compat.v1.Session(config=self.config)

    def build_autoencoder(self) -> None:
        """Builds autoencoder architecture and assigns to instance attributes.

        Returns:
            None, instantiates encoder & autoencoder attributes
        """
        raise NotImplementedError

    def fit(self,
            X_train,
            X_test,
            num_epochs,
            batch_size,
            encoder_only=False,
            data_aug=True,
            callbacks: Union[Tuple[callable], None] = None) -> None:
        """Fits model (encoder/autoencoder) to provided data.

        Args:
            X_test (numpy Array):
            X_train (numpy Array):
            num_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            encoder_only (bool): Flag to use encoder only instead of fitting whole autoencoder.

        Returns:
             None, trains encoder/autoencoder attribute
        """

        if data_aug:
            self.NAME = self.NAME + '_datagen'

            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                featurewise_center=False,
                featurewise_std_normalization=False,
                width_shift_range=0.20,
                height_shift_range=0.20,
                horizontal_flip=True
            )
            datagen.fit(X_train)

            if with_tensorboard:
                tb_callback = TensorBoard(log_dir='../logs/{}'.format(self.NAME), update_freq='epoch')

                self.autoencoder.fit(datagen.flow(X_train, X_train, batch_size=batch_size_),
                    epochs=num_epochs,
                    steps_per_epoch=len(X_train)/batch_size_,
                    validation_data=(X_test,X_test),
                    callbacks=[tb_callback])
            else:
                self.autoencoder.fit(datagen.flow(X_train, X_train, batch_size=batch_size_),
                    epochs=num_epochs,
                    steps_per_epoch=len(X_train)/batch_size_,
                    validation_data=(X_test,X_test))
        else:
            if with_tensorboard:
                tb_callback = TensorBoard(log_dir='../logs/{}'.format(self.NAME), update_freq='epoch')

                self.autoencoder.fit(X_train, X_train,
                    epochs=num_epochs,
                    batch_size=batch_size_,
                    validation_data=(X_test,X_test),
                    callbacks=[tb_callback])
            else:
                self.autoencoder.fit(X_train, X_train,
                    epochs=num_epochs,
                    batch_size=batch_size_,
                    validation_data=(X_test,X_test))

        # self._extract_latent(X_test)

    def kmean_cluster(self, latents, num_clusters, set_seed=True):
        '''
        Cluster encoded images to means
        '''
        if set_seed:
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=33)
        else:
            self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(latents)
    
    def elbow_plot(self, latents, max_k, f_name):
        '''
        Plot elbow plot
        '''
        fig, ax = plt.subplots(1, figsize=(12,12))

        rss_lst = []
        for k in range(1, max_k):
            self.kmean_cluster(latents, k)
            rss_lst.append(self.kmeans.inertia_)
        
        ax.plot(range(1, max_k), rss_lst)
        ax.set_ylabel('RSS')
        ax.set_xlabel('Number of Clusters')
        ax.set_title('Elbow Plot for Model:\n{}'.format(f_name))

        self._save_fig('elbow_{}.png'.format(f_name))

    def top_9_from_clusters(self, X_test, latent, model_name, gray_imgs=True):
        '''
        Plot top 9 from each cluster in 3x3 grid
        '''
        centers = self.kmeans.cluster_centers_
        cluster_labels = self.kmeans.labels_

        tops = {}
        distances = {}

        for label in np.unique(cluster_labels)[::1]:
            dist = cosine_distances(latent[np.where(cluster_labels == label)], centers[label,:].reshape(1,-1))
            top_dist = dist[np.argsort(dist.T)[::-1][0][:9]]
            top_ = X_test[np.argsort(dist.T)[::-1][0][:9]]
            tops[label] = top_
            distances[label] = top_dist
        # pdb.set_trace()
        
        for label in np.unique(cluster_labels)[::1]:
            fig, axes = plt.subplots(3,3,figsize=(12,12))
            for ax, img in zip(axes.flatten(), tops[label]):
                if gray_imgs:
                    ax.imshow(img.squeeze(), cmap='gray')
                else:
                    ax.imshow(img)
                ax.set_axis_off()
            fig.suptitle('Top 9: Cluster {}\n Model: {}'.format(label, model_name))
            self._save_fig('clusters/top9_cluster_{}_{}.png'.format(label, model_name))

    def save_model(self,model_name):
        '''
        Method to save model and latent features
        '''
        self.autoencoder.save('../models/{}_{}_{}'.format(
            model_name, str(datetime.now().date()), str(datetime.now().time())
        ))

        with open('../models/{}_{}_{}_xtest_encode.pkl'.format(
            model_name, str(datetime.now().date()), str(datetime.now().time())
        ), 'wb') as f:
            pickle.dump(self.latent,f)

    def load_model(self,model_fname, latent_fname):
        '''
        Load previously saved model and latent features.
        '''
        self._clear_variables()

        self.autoencoder = keras.models.load_model('../models/{}'.format(model_fname))
        with open('../models/{}'.format(latent_fname), 'rb') as f:
            self.latent = pickle.load(f)
    
    def _save_fig(self, file_name):
        '''
        Saves and closes figure.
        '''
        plt.savefig('../images/{}'.format(file_name), dpi=100)
        plt.close('all')

    def _extract_latent(self, X_test):
        '''
        Extract encoded latent features
        '''
        print('\nShape of X:{}\n'.format(X_test.shape))
        batches = np.split(X_test,X_test.shape[0])           # 95 for test, 199 for holdout/all
        for i,batch in enumerate(batches):
            get_layer_output = K.function([self.autoencoder.layers[0].input],
                                        [self.autoencoder.layers[13].output])
            # pdb.set_trace()
            layer_output = get_layer_output([batch])[0]

            if i == 0:
                final_layers = layer_output
            else:
                final_layers = np.vstack((final_layers,layer_output))
        self.latent = final_layers

    def load_Xy(self, date, with_href=True):
        '''
        Load Xy matrices to use without calling pipeline.
        '''
        if with_href:
            if self.gray_imgs:
                X_fname = '../data/Xs/gray_{}_original'.format(date)
                with open(X_fname, 'rb') as f:
                    self.X_total = pickle.load(f)
                y_fname = '../data/ys/gray_{}_href'.format(date)
                with open(y_fname, 'rb') as f:
                    self.y_href = pickle.load(f)
            else:
                X_fname = '../data/Xs/rgb_{}_original'.format(date)
                with open(X_fname, 'rb') as f:
                    self.X_total = pickle.load(f)
                y_fname = '../data/ys/rgb_{}_href'.format(date)
                with open(y_fname, 'rb') as f:
                    self.y_href = pickle.load(f)

        else:
            if self.gray_imgs:
                X_fname = '../data/Xs/gray_{}'.format(date)
                with open(X_fname, 'rb') as f:
                    self.X_gray = pickle.load(f)
                y_fname = '../data/ys/gray_{}'.format(date)
                with open(y_fname, 'rb') as f:
                    self.y_gray = pickle.load(f)
            else: 
                X_fname = '../data/Xs/rgb_{}'.format(date)
                with open(X_fname, 'rb') as f:
                    self.X_rgb = pickle.load(f)
                y_fname = '../data/ys/rgb_{}'.format(date)
                with open(y_fname, 'rb') as f:
                    self.y_rgb = pickle.load(f)


class GalvanizeAutoencoder(BaseAutoencoder):
    def __init__(self, *args, **kwargs):
        super(GalvanizeAutoencoder, self).__init__(*args, **kwargs)

    def build_autoencoder(self,
                          init_num_filters: int = 64,
                          num_encode_layers: int = 4,
                          image_shape: Tuple[int, int, int] = (128, 128, 1),
                          drop_out: float = 0.5,
                          max_norm_constraint: int = 2,
                          kernel_size: Tuple[int, int] = (3, 3)) -> None:
        """Builds autoencoder using architecture developed at Galvanize.

        Args:
            init_num_filters (int): Initial number of filters for the first CONV layer
            num_encode_layers (int): Number of CONV layers in encoder.
            image_shape (tuple (int, int, int)): Shape of images to be used.
                grayscale images should have a 3rd dim of 1, while RGB have a 3rd dim of 3
            drop_out (float): Fraction of neurons to drop after each layer.
            max_norm_constraint (int): Max Norm constraint for CONV2D layer.
            kernel_size (tuple (int, int)): Size of CONV kernel.

        Returns:
            None, assigns encoder/autoencoder attributes.
        """
        encoder_input = keras.Input(shape=image_shape, name="image")
        out_filter = image_shape[-1]

        for idx, layer in enumerate(range(num_encode_layers, -1, -1)):
            if idx:
                previous_layer = x
            else:
                previous_layer = encoder_input

            x = layers.Conv2D(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                activation='relu',
                kernel_constraint=max_norm(max_norm_constraint, axis=[0, 1, 2])
            )(previous_layer)

            x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            x = layers.SpatialDropout2D(rate=drop_out)(x)
        encoder_output = layers.Flatten()(x)
        self.encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        resize_size = image_shape[0] // (2**(num_encode_layers + 1))
        resize_layers = int(init_num_filters)

        x = layers.Reshape(target_shape=(resize_size, resize_size, resize_layers))(encoder_output)

        for layer in range(num_encode_layers + 1):
            x = layers.Conv2DTranspose(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                activation='relu',
                kernel_constraint=max_norm(max_norm_constraint, axis=[0, 1, 2])
            )(x)
            x = layers.UpSampling2D(size=(2, 2))(x)
            x = layers.SpatialDropout2D(rate=drop_out)(x)
        decoder_output = layers.Conv2DTranspose(
            filters=out_filter,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            activation='sigmoid'
        )(x)
        self.autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')


class XceptionAutoencoder(BaseAutoencoder):
    def __init__(self, *args, **kwargs):
        super(XceptionAutoencoder, self).__init__(*args, **kwargs)

    def build_autoencoder(self,
                          init_num_filters: int = 64,
                          num_encode_layers: int = 4,
                          image_shape: Tuple[int, int, int] = (128, 128, 1),
                          drop_out: float = 0.5,
                          max_norm_constraint: int = 2,
                          kernel_size: Tuple[int, int] = (3, 3)) -> None:
        """Builds autoencoder using architecture adapted from Xception model Head.

        Args:
            init_num_filters (int): Initial number of filters for the first CONV layer
            num_encode_layers (int): Number of CONV layers in encoder.
            image_shape (tuple (int, int, int)): Shape of images to be used.
                grayscale images should have a 3rd dim of 1, while RGB have a 3rd dim of 3
            drop_out (float): Fraction of neurons to drop after each layer.
            max_norm_constraint (int): Max Norm constraint for CONV2D layer.
            kernel_size (tuple (int, int)): Size of CONV kernel.

        Returns:
            None, assigns encoder/autoencoder attributes.
        """
        encoder_input = keras.Input(shape=image_shape, name="image")
        out_filter = image_shape[-1]

        for idx, layer in enumerate(range(num_encode_layers, -1, -1)):
            if idx:
                previous_layer = x
            else:
                previous_layer = encoder_input

            x = layers.SeparableConv2D(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                padding='same'
            )(previous_layer)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(pool_size=2, padding='same')(x)
        encoder_output = layers.Flatten()(x)
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')

        resize_size = image_shape[0] // (2**(num_encode_layers + 1))
        resize_layers = int(init_num_filters)

        x = layers.Reshape(target_shape=(resize_size, resize_size, resize_layers))(encoder_output)

        for layer in range(num_encode_layers + 1):
            x = layers.Conv2DTranspose(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2DTranspose(
                filters=(init_num_filters // (2**layer)),
                kernel_size=kernel_size,
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.UpSampling2D(size=(2, 2))(x)
        decoder_output = layers.Conv2DTranspose(
            filters=out_filter,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            activation='sigmoid'
        )(x)

        self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
        self.autoencoder.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )


class BaseRecommender:
    def __init__(self, distance_function: callable = cosine_distances):
        self.distance_function = distance_function


if __name__ == "__main__":
    model = GalvanizeAutoencoder()
    model.build_autoencoder()
    print(model.autoencoder.summary())

    model = XceptionAutoencoder()
    model.build_autoencoder()
    print(model.autoencoder.summary())

