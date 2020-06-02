# conventional import
import numpy as np

# tf imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Autoencoder():
    '''
    This class will build CNN autoencoder.
    '''
    def __init__(self):
        self._clear_variables()

    def _clear_variables(self):
        '''
        Set attributes to empty type variables
        '''
        self.encoded_images = []
        self.encoder = None

    def build_autoencoder(self, init_num_filters, num_encode_layers, use_color_img=True):
        '''
        Functional API build of model
        '''
        if use_color_img:
            inputs = keras.Input(shape=(128,128,3))
        else:
            inputs = keras.Input(shape=(128,128,1))
            
        layer_list = []
        for encode_layer in range(num_encode_layers):
            if encode_layer == 0:
                encode_1 = keras.layers.Conv2D(
                    filters=init_num_filters,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='relu',
                    use_bias=True
                )(inputs)

                layer_list.append(encode_1)
                
                layer_list.append(
                    keras.layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[0])
                )

            else:
                layer_list.append(
                    keras.layers.Conv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                    )(layer_list[2*encode_layer-1])
                )

                layer_list.append(
                    keras.layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[2*encode_layer])
                )
        
        # conv dropout
        layer_list.append(
            keras.layers.SpatialDropout2D(
                rate=0.5
            )(layer_list[2*num_encode_layers-1])
        )

        self.encoder = keras.Model(inputs, layer_list[-1])

        self.latent = keras.layers.Flatten()(layer_list[-1])

        

        



if __name__ == "__main__":
    model = Autoencoder()
    model.build_autoencoder(128,5)
    encoder = model.encoder
    print(encoder.summary())