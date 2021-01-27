# conventional import
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})
import pdb

# tf imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.constraints import max_norm

# sk imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

# import base class from cnn_new
from cnn_new import Autoencoder


class XceptionAE(Autoencoder):
    def __init__(self, gray_imgs=True):
        super().__init__(gray_imgs=gray_imgs)

    def build_autoencoder(self, init_num_filters, num_encode_layers, enc_do=0.5, dec_do=0.5, max_norm_value=2, kernel_size=(3,3)):
        '''
        Functional API build of model
        input shape = (128,128,x) where x=1,3 1=greyscale
        num encode/decode layers = 5

        128 init layers --> 5 encoding layers for 128 feats

        '''
        self.init_num_filters = init_num_filters
        self.num_encode_layers = num_encode_layers
        self.kernel_size = kernel_size

        if self.gray_imgs:
            inputs = keras.Input(shape=(128,128,1))
            out_filter = 1
        else:
            inputs = keras.Input(shape=(128,128,3))
            out_filter = 3
            
        layer_list = []
        for encode_layer in range(num_encode_layers)[::-1]:
            if encode_layer == max(range(num_encode_layers)):
                layer_list.append(
                    layers.SeparableConv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=kernel_size,
                        padding='same'
                    )(inputs)
                )
                
                layer_list.append(
                    layers.BatchNormalization()(layer_list[-1])
                )

                layer_list.append(
                    layers.Activation('relu')(layer_list[-1])
                )

                layer_list.append(
                    layers.SeparableConv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=kernel_size,
                        padding='same'
                    )(layer_list[-1])
                )

                layer_list.append(
                    layers.BatchNormalization()(layer_list[-1])
                )

                layer_list.append(
                    layers.Activation('relu')(layer_list[-1])
                )

                layer_list.append(
                    layers.MaxPooling2D(
                        pool_size=2,
                        padding='same'
                    )(layer_list[-1])
                )

            else:
                layer_list.append(
                    layers.SeparableConv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=kernel_size,
                        padding='same'
                    )(layer_list[-1])
                )
                
                layer_list.append(
                    layers.BatchNormalization()(layer_list[-1])
                )

                layer_list.append(
                    layers.Activation('relu')(layer_list[-1])
                )

                layer_list.append(
                    layers.SeparableConv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=kernel_size,
                        padding='same'
                    )(layer_list[-1])
                )

                layer_list.append(
                    layers.BatchNormalization()(layer_list[-1])
                )

                layer_list.append(
                    layers.Activation('relu')(layer_list[-1])
                )

                layer_list.append(
                    layers.MaxPooling2D(
                        pool_size=2,
                        padding='same'
                    )(layer_list[-1])
                )

        layer_list.append(
            layers.Flatten()(layer_list[-1])
        )

        self.encoder = keras.Model(inputs, layer_list[-1])


        resize_side = int(128/(2**num_encode_layers))
        resize_layers = int(init_num_filters)

        layer_list.append(
            layers.Reshape(
                target_shape=(resize_side,resize_side,resize_layers)
            )(layer_list[-1])
        )

        for decode_layer in range(num_encode_layers):
            layer_list.append(
                layers.Conv2DTranspose(
                    filters=(init_num_filters // (2**decode_layer)),
                    kernel_size=kernel_size,
                    padding='same'
                )(layer_list[-1])
            )

            layer_list.append(
                layers.BatchNormalization()(layer_list[-1])
            )

            layer_list.append(
                layers.Activation('relu')(layer_list[-1])
            )

            layer_list.append(
                layers.Conv2DTranspose(
                    filters=(init_num_filters // (2**decode_layer)),
                    kernel_size=kernel_size,
                    padding='same'
                )(layer_list[-1])
            )
            
            layer_list.append(
                layers.BatchNormalization()(layer_list[-1])
            )

            layer_list.append(
                layers.Activation('relu')(layer_list[-1])
            )

            layer_list.append(
                layers.UpSampling2D(
                    size=(2,2)
                )(layer_list[-1])
            )


        layer_list.append(
            layers.Conv2DTranspose(
                    filters=out_filter,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='sigmoid'
            )(layer_list[-1])
        )

        self.autoencoder = keras.Model(inputs, layer_list[-1])
        self.autoencoder.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

if __name__ == "__main__":
    model_builder = XceptionAE(gray_imgs=False)
    model_builder.build_autoencoder(64, 5)
    
    model = model_builder.autoencoder
    print(model.summary())