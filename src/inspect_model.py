# general imports
import pickle
import numpy as np
import pdb

import tensorflow as tf

# other file imports
from cnn_new import Autoencoder
from data_pipelines import ImagePipeline
from plotting import plot_before_after, cluster_plot, silhouette


if __name__ == "__main__":
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    # model names updated on friday
    gray_fname = 'ae_og_convT_gray_15eps_10batch_64initfilts_4layers_128img__50do_2norm_3kernel_2020-06-08_17:51:39.718211_datagen_2020-06-08_17:53:21.290709'
    rgb_fname = 'ae_og_convT_color_15eps_10batch_64initfilts_4layers_128img__50do_2norm_3kernel_2020-06-08_17:45:47.036921_datagen_2020-06-08_17:47:43.071002'

    # latent fnames
    gray_latent_fname = 'ae_og_convT_gray_10eps_10batch_128initfilts_5layers_128img__50do_2norm_3kernel_2020-06-08_09:29:54.041637_datagen_2020-06-08_09:35:23.915315_xtest_encode.pkl'
    rgb_latent_fname = 'ae_og_convT_color_50eps_10batch_128initfilts_4layers_128img__50do_2norm_3kernel_2020-06-08_14:35:16.894755_datagen_2020-06-08_14:46:47.676980_xtest_encode.pkl'

    # load data

    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    rgb.load_Xy('2020-06-04')
    gray.load_Xy('2020-06-04')
    
    X_rgb = rgb.X_rgb
    X_gray = gray.X_gray

    rgb.load_model(rgb_fname, rgb_latent_fname)
    gray.load_model(gray_fname, gray_latent_fname)

    '''
    Elbow plots
    '''
    # concat both feats for kmeans
    # plot rgb/ gray only elbows
    # rgb.elbow_plot(rgb.latent,20,'Color Only')
    # gray.elbow_plot(gray.latent,20,'Gray Only')

    # combo = np.hstack((rgb.latent,gray.latent))

    # rgb.elbow_plot(combo,20,'Ensemble')

    '''
    top 9 imgs
    '''
    X_gray_holdout = X_gray['holdout'].reshape(X_gray['holdout'].shape[0], 128, 128, 1)
    X_rgb_holdout = X_rgb['holdout'].reshape(X_rgb['holdout'].shape[0], 128, 128, 3)


    X_gray_holdout = X_gray_holdout.astype('float32') 
    X_rgb_holdout = X_rgb_holdout.astype('float32') 

    X_gray_holdout = X_gray_holdout/255
    X_rgb_holdout = X_rgb_holdout/255

    X_predict = rgb.autoencoder.predict(X_rgb_holdout)

    # plot_before_after(X_rgb_holdout, X_predict, 'color 64x4')

    # gray.kmean_cluster(gray.latent,7,set_seed=False)
    # gray._extract_latent(X_gray_holdout)
    # cluster_plot(X_rgb_holdout, gray.kmeans.labels_,'gray')

    rgb._extract_latent(X_rgb_holdout)
    rgb.kmean_cluster(rgb.latent,7,set_seed=False)


    '''
    1393/1393 [==============================] - 4s 3ms/sample - loss: 0.0404       RGB
    1393/1393 [==============================] - 3s 2ms/sample - loss: 0.0366       Gray
    '''