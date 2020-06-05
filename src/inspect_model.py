# general imports
import pickle
import numpy as np
import pdb

# other file imports
from cnn_new import Autoencoder
from data_pipelines import ImagePipeline

if __name__ == "__main__":
    # model names
    gray_fname = 'new_ae_gray_30eps_5batch_128_5down5up_50do_128feats_listings_datagen_2020-06-04_15:25:12.077030'
    rgb_fname = 'new_ae_color_60eps_5batch_128_5down5up_50do_128feats_listings_datagen_2020-06-04_14:29:49.502914'

    # latent fnames
    gray_latent_fname = 'new_ae_gray_30eps_5batch_128_5down5up_50do_128feats_listings_datagen_2020-06-04_15:25:12.521068_xtest_encode.pkl'
    rgb_latent_fname = 'new_ae_color_60eps_5batch_128_5down5up_50do_128feats_listings_datagen_2020-06-04_14:29:49.973571_xtest_encode.pkl'

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
    # rgb.elbow_plot(rgb.latent,20,'rgb_test')
    # gray.elbow_plot(gray.latent,20,'gray_test')

    combo = np.hstack((rgb.latent,gray.latent))

    # rgb.elbow_plot(combo,20,'combo_test')

    '''
    top 9 imgs
    '''
    X_gray_test = X_gray['test'].reshape(X_gray['test'].shape[0], 128, 128, 1)
    X_rgb_test = X_rgb['test'].reshape(X_rgb['test'].shape[0], 128, 128, 3)

    X_gray_test = X_gray_test.astype('float32') 
    X_rgb_test = X_rgb_test.astype('float32') 

    X_gray_test = X_gray_test/255
    X_rgb_test = X_rgb_test/255    

    gray.kmean_cluster(gray.latent,7)
    rgb.kmean_cluster(rgb.latent,7)

    gray.top_9_from_clusters(X_rgb_test,gray_fname)
    rgb.top_9_from_clusters(X_rgb_test, rgb_fname)
