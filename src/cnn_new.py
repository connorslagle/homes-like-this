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

    def build_autoencoder(self, use_color_img=True):
        '''
        Functional API build of model
        input shape = (128,128,x) where x=1,3 1=greyscale
        num encode/decode layers = 5
        '''
        init_num_filters = 128
        num_encode_layers = 5

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
                    )(layer_list[-1])
                )

            else:
                layer_list.append(
                    keras.layers.Conv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='relu'
                    )(layer_list[-1])
                )

                layer_list.append(
                    keras.layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[-1])
                )
        

        layer_list.append(
            keras.layers.SpatialDropout2D(
                rate=0.5
            )(layer_list[-1])
        )

        layer_list.append(
            keras.layers.Flatten()(layer_list[-1])
        )

        self.encoder = keras.Model(inputs, layer_list[-1])

        self.latent = self.encoder.output

        layer_list.append(
            keras.layers.Reshape(
                target_shape=(4,4,8)
            )(layer_list[-1])
        )

        for decode_layer in range(num_encode_layers)[::-1]:
            layer_list.append(
                keras.layers.Conv2D(
                    filters=(init_num_filters // (2**decode_layer)),
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='relu'
                )(layer_list[-1])
            )

            layer_list.append(
                keras.layers.UpSampling2D(
                    size=(2,2)
                )(layer_list[-1])
            )

        layer_list.append(
            keras.layers.Conv2D(
                    filters=3,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='sigmoid'
            )(layer_list[-1])
        )

        self.encoder_decoder = keras.Model(inputs, layer_list[-1])
        self.encoder_decoder.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )



if __name__ == "__main__":
    model = Autoencoder()
    model.build_autoencoder()
    model1 = model.encoder_decoder
    print(model1.summary())