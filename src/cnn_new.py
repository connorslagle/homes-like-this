# conventional import
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

# tf imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

# sk imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances


class Autoencoder():
    def __init__(self):
        '''
        This class will build CNN autoencoder.
        '''
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
            out_filter = 3
        else:
            inputs = keras.Input(shape=(128,128,1))
            out_filter = 1
            
        layer_list = []
        for encode_layer in range(num_encode_layers):
            if encode_layer == 0:
                encode_1 = layers.Conv2D(
                    filters=init_num_filters,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='relu',
                    use_bias=True
                )(inputs)

                layer_list.append(encode_1)
                
                layer_list.append(
                    layers.MaxPooling2D(
                        pool_size=(2,2),
                        padding='same'
                    )(layer_list[-1])
                )

            else:
                layer_list.append(
                    layers.Conv2D(
                        filters=(init_num_filters // (2**encode_layer)),
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='relu'
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
                rate=0.5
            )(layer_list[-1])
        )

        layer_list.append(
            layers.Flatten()(layer_list[-1])
        )

        self.encoder = keras.Model(inputs, layer_list[-1])

        self.latent = self.encoder.output

        layer_list.append(
            layers.Reshape(
                target_shape=(4,4,8)
            )(layer_list[-1])
        )

        for decode_layer in range(num_encode_layers)[::-1]:
            layer_list.append(
                layers.Conv2D(
                    filters=(init_num_filters // (2**decode_layer)),
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='relu'
                )(layer_list[-1])
            )

            layer_list.append(
                layers.UpSampling2D(
                    size=(2,2)
                )(layer_list[-1])
            )

        layer_list.append(
            layers.Conv2D(
                    filters=out_filter,
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

    def fit_(self, X_train, X_test, num_epochs, batch_size_, use_gpu=True, with_tensorboard=True):
        '''
        Fits Autoencoder to data
        '''
        self.NAME = "new_ae_{}eps_{}batch_128_5down5up_50do_128feats_listings".format(
            num_epochs, batch_size_
        )

        if use_gpu:
            self._use_gpu()

        if with_tensorboard:
            self.encoder_decoder.fit(X_train, X_train,
                epochs=num_epochs,
                batch_size=batch_size_,
                validation_data=(X_test,X_test),
                callbacks=TensorBoard(log_dir='../logs/{}'.format(self.NAME)))
        else:
            self.encoder_decoder.fit(X_train, X_train,
                epochs=num_epochs,
                batch_size=batch_size_,
                validation_data=(X_test,X_test))
            
    def kmean_cluster(self, X_test, num_clusters):
        '''
        Cluster encoded images to means
        '''
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(X_test)
    
    def elbow_plot(self, X_test, max_k, f_name):
        '''
        Plot elbow plot
        '''
        fig, ax = plt.subplots(1, figsize=(12,12))

        rss_lst = []
        for k in range(1, max_k):
            self.kmean_cluster(X_test, k)
            rss_lst.append(self.kmeans.inertia_)
        
        ax.plot(range(1, max_k), rss_lst)
        ax.set_ylabel('RSS')
        ax.set_xlabel('Number of Clusters')
        ax.set_title('Elbow Plot for Model:\n{}'.format(self.NAME))

        self._save_fig('elbow_{}.png'.format(self.NAME))

    def top_9_from_clusters(self, X_test):
        '''
        Plot top 9 from each cluster in 3x3 grid
        '''
        centers = self.kmeans.cluster_centers_
        cluster_labels = self.kmeans.labels_

        tops = {}
        distances = {}

        for label in np.unique(cluster_labels)[::1]:
            dist = cosine_distances(layers[np.where(cluster_labels == label)], centers[label,:].reshape(1,-1))
            top_dist = dist[np.argsort(dist.T)[::-1][0][:9]]
            top_ = X_test[np.argsort(dist.T)[::-1][0][:9]]
            tops[label] = top_
            distances[label] = top_dist
        
        fig, axes = plt.subplots(3,3,figsize=(12,12))
        for label in np.unique(cluster_labels)[::1]:
            for ax, img in zip(axes.flatten(), tops[label]):
                ax.imshow(img)
                ax.set_axis_off()
            fig.suptitle('Top 9: Cluster {}\n Model: {}'.format(label, self.NAME))
            self._save_fig('top9_cluster_{}_{}.png'.format(label, self.NAME))

    def silhouette_plot(self):
        '''
        Plots silhouette plot, fill in later.
        '''
        pass

    def save_model(self):
        '''
        Method to save model and latent features
        '''
        self.encoder_decoder.save('../models/{}_{}_{}'.format(
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

        self.encoder_decoder = keras.models.load_model('../models/{}'.format(model_fname))
        with open('../models/{}'.format(latent_fname), 'rb') as f:
            self.latent = pickle.load(f)
    
    def _save_fig(self, file_name):
        '''
        Saves and closes figure.
        '''
        plt.savefig('../images/{}'.format(file_name), dpi=100)
        plt.close('all')


if __name__ == "__main__":
    model = Autoencoder()
    model.build_autoencoder(use_color_img=False)
    model1 = model.encoder_decoder
    print(model1.summary())