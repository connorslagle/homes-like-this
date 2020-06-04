# general imports
import pickle

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

    X_test_gray = gray_pipe.X_test


    X_test_rgb = rgb_pipe.X_test

    # load model/latents

    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    X_test_gray = 

    rgb.load_model(rgb_fname, rgb_latent_fname)
    gray.load_model(gray_fname, gray_latent_fname)

    # concat both feats for kmeans
    
    
