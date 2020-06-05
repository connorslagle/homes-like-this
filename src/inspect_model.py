# general imports
import pickle
import numpy as np
import pdb

import tensorflow as tf

# other file imports
from cnn_new import Autoencoder
from data_pipelines import ImagePipeline

if __name__ == "__main__":
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    # model names updated on friday
    gray_fname = 'models/rand_ae_convT_gray_3eps_128batch_128_5down5up_50do_2norm_128feats_2020-06-05_11:49:49.291655_datagen_2020-06-05_11:53:46.657751'
    rgb_fname = 'models/rand_ae_convT_color_5eps_128batch_128_5down5up_50do_2norm_128feats_2020-06-05_11:56:06.697486_datagen_2020-06-05_12:00:55.866913'

    # latent fnames
    gray_latent_fname = 'models/rand_ae_convT_gray_3eps_128batch_128_5down5up_50do_2norm_128feats_2020-06-05_11:49:49.291655_datagen_2020-06-05_11:53:47.323588_xtest_encode.pkl'
    rgb_latent_fname = 'models/rand_ae_convT_color_5eps_128batch_128_5down5up_50do_2norm_128feats_2020-06-05_11:56:06.697486_datagen_2020-06-05_12:00:56.525856_xtest_encode.pkl'

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
    # rgb.elbow_plot(rgb.latent,20,'Color_Only')
    # gray.elbow_plot(gray.latent,20,'Gray_Only')

    combo = np.hstack((rgb.latent,gray.latent))

    # rgb.elbow_plot(combo,20,'Ensemble')

    '''
    top 9 imgs
    '''
    # X_gray_holdout = X_gray['holdout'].reshape(X_gray['holdout'].shape[0], 128, 128, 1)
    X_rgb_holdout = X_rgb['holdout'].reshape(X_rgb['holdout'].shape[0], 128, 128, 3)


    # X_gray_test = X_gray_test.astype('float32') 
    X_rgb_holdout = X_rgb_holdout.astype('float32') 

    # X_gray_test = X_gray_test/255
    X_rgb_holdout = X_rgb_holdout/255    

    # gray.kmean_cluster(gray.latent,7)
    rgb._extract_latent(X_rgb_holdout)
    rgb.kmean_cluster(rgb.latent,7,set_seed=False)
    rgb.top_9_from_clusters(X_rgb_holdout,rgb.latent, 'RGB Latent Holdout')

    gray._extract_latent(X_rgb_holdout)
    rgb.kmean_cluster(gray.latent,7,set_seed=False)
    rgb.top_9_from_clusters(X_rgb_holdout, gray.latent, 'Gray Latent Holdout')

    combo = np.hstack((rgb.latent,gray.latent))

    rgb.kmean_cluster(combo,7,set_seed=False)
    rgb.top_9_from_clusters(X_rgb_holdout, combo, 'Ensemble Latent Holdout')

    # gray.top_9_from_clusters(X_rgb_test,gray_fname)
    # rgb.top_9_from_clusters(X_rgb_test, rgb_fname)
