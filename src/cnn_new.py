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


class Autoencoder():
    def __init__(self, gray_imgs=True):
        '''
        This class will build CNN autoencoder.
        '''
        self.gray_imgs = gray_imgs
        self._clear_variables()

    def _clear_variables(self):
        '''
        Set attributes to empty type variables
        '''
        self.encoded_images = []
        self.encoder = None
        self.latent = None
        self.encoder_decoder = None
        self.config = None

    def _use_gpu(self):
        '''
        Sets tf environment for gpu operation
        '''
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        tf.compat.v1.Session(config=self.config)

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
        for encode_layer in range(num_encode_layers):
            if encode_layer == 0:
                encode_1 = layers.Conv2D(
                    filters=init_num_filters,
                    kernel_size=kernel_size,
                    strides=(1,1),
                    padding='same',
                    activation='relu',
                    use_bias=True,
                    kernel_constraint=max_norm(max_norm_value, axis=[0,1,2])
                )(inputs)

                layer_list.append(encode_1)
                
                layer_list.append(
                    layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[-1])
                )

                layer_list.append(
                    layers.SpatialDropout2D(
                        rate=enc_do
                    )(layer_list[-1])
                )

            else:
                layer_list.append(
                    layers.Conv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=kernel_size,
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                        kernel_constraint=max_norm(max_norm_value, axis=[0,1,2])
                    )(layer_list[-1])
                )

                layer_list.append(
                    layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[-1])
                )

                layer_list.append(
                    layers.SpatialDropout2D(
                        rate=enc_do
                    )(layer_list[-1])
                )

        layer_list.append(
            layers.Flatten()(layer_list[-1])
        )

        self.encoder = keras.Model(inputs, layer_list[-1])


        resize_side = int(128/(2**num_encode_layers))
        resize_layers = int(init_num_filters/(2**(num_encode_layers-1)))

        layer_list.append(
            layers.Reshape(
                target_shape=(resize_side,resize_side,resize_layers)
            )(layer_list[-1])
        )

        for decode_layer in range(num_encode_layers)[::-1]:
            layer_list.append(
                layers.Conv2DTranspose(
                    filters=(init_num_filters // (2**decode_layer)),
                    kernel_size=kernel_size,
                    strides=(1,1),
                    padding='same',
                    activation='relu',
                    kernel_constraint=max_norm(max_norm_value, axis=[0,1,2])
                )(layer_list[-1])
            )

            layer_list.append(
                layers.UpSampling2D(
                    size=(2,2)
                )(layer_list[-1])
            )

            layer_list.append(
                layers.SpatialDropout2D(
                    rate=dec_do
                )(layer_list[-1])
            )

        layer_list.append(
            layers.Conv2DTranspose(
                    filters=out_filter,
                    kernel_size=kernel_size,
                    strides=(1,1),
                    padding='same',
                    activation='sigmoid'
            )(layer_list[-1])
        )

        self.autoencoder = keras.Model(inputs, layer_list[-1])
        self.autoencoder.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

    def fit_(self, X_train, X_test, num_epochs, batch_size_, use_gpu=True, data_aug=True, with_tensorboard=True):
        '''
        Fits Autoencoder to data
        '''
        if self.gray_imgs:
            self.NAME = "ae_og_convT_{}_{}eps_{}batch_{}initfilts_{}layers_128img__50do_2norm_{}kernel_{}_{}".format(
                'gray', num_epochs, batch_size_, self.init_num_filters, self.num_encode_layers, self.kernel_size[0] \
                        , str(datetime.now().date()), str(datetime.now().time())
            )
        else:
            self.NAME = "ae_og_convT_{}_{}eps_{}batch_{}initfilts_{}layers_128img__50do_2norm_{}kernel_{}_{}".format(
                'color', num_epochs, batch_size_, self.init_num_filters, self.num_encode_layers, self.kernel_size[0] \
                        , str(datetime.now().date()), str(datetime.now().time())
            )


        if use_gpu:
            self._use_gpu()

        if data_aug:
            self.NAME = self.NAME + '_datagen'

            datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
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

        self._extract_latent(X_test)
            
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

    def silhouette_plot(self):
        '''
        Plots silhouette plot, fill in later.
        '''
        pass

    def save_model(self):
        '''
        Method to save model and latent features
        '''
        self.autoencoder.save('../models/{}_{}_{}'.format(
            self.NAME, str(datetime.now().date()), str(datetime.now().time())
        ))

        with open('../models/{}_{}_{}_xtest_encode.pkl'.format(
            self.NAME, str(datetime.now().date()), str(datetime.now().time())
        ), 'wb') as f:
            pickle.dump(self.latent,f)

    def load_model(self,model_fname, latent_fname):
        '''
        Load previously saved model and latent features.
        '''
        self._clear_variables()

        self.autoencoder = keras.models.load_model('../models_aws/models/{}'.format(model_fname))
        with open('../models_aws/models/{}'.format(latent_fname), 'rb') as f:
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
        batches = np.split(X_test,95)
        for i,batch in enumerate(batches):
            get_layer_output = K.function([self.autoencoder.layers[0].input],
                                        [self.autoencoder.layers[16].output])
            layer_output = get_layer_output([batch])[0]

            if i == 0:
                final_layers = layer_output
            else:
                final_layers = np.vstack((final_layers,layer_output))
        self.latent = final_layers

    def load_Xy(self, date):
        '''
        Load Xy matrices to use without calling pipeline.
        '''
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



if __name__ == "__main__":
    model = Autoencoder(gray_imgs=True)
    model.build_autoencoder(128,5)
    model1 = model.autoencoder
    print(model1.summary())