#general imports
import numpy as np
import pandas as pd
import pdb
import pickle
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})

# docker connected
import pymongo
import tensorflow as tf

# sklearn
from sklearn.metrics.pairwise import cosine_similarity


# from other py files
from data_pipelines import ImagePipeline, MongoImporter
from cnn_new import Autoencoder
from plotting import img_plot_3x3


if __name__ == "__main__":
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    '''
    Import models and img latents
    '''

    # model names updated on friday
    gray_fname = 'gray_final_2020-06-09_17:52:27.380764'
    rgb_fname = 'rgb_final_2020-06-09_17:52:23.902556'

    # latent fnames
    gray_latent_fname = 'gray_final_2020-06-09_17:52:30.434822_xtest_encode.pkl'
    rgb_latent_fname = 'rgb_final_2020-06-09_17:52:27.223554_xtest_encode.pkl'

    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    rgb.load_model(rgb_fname, rgb_latent_fname)
    gray.load_model(gray_fname, gray_latent_fname)

    rgb.load_Xy('2020-06-04', with_href=False)
    gray.load_Xy('2020-06-04', with_href=False)
    
    X_rgb = rgb.X_rgb
    X_gray = gray.X_gray

    X_total = np.vstack((X_rgb['train'], X_rgb['test'], X_rgb['holdout']))

    '''
    Load img and data pipelines
    '''
    color_pipe = ImagePipeline('../data/test_imgs/raw',gray_imgs=False)
    color_pipe.read()
    color_pipe._square_image()
    color_pipe._resize((128,128))
    color_pipe.save()

    color_pipe.vectorize()

    X_user_color = color_pipe.features
    y_user_color = color_pipe.labels

    gray_pipe = ImagePipeline('../data/test_imgs/raw',gray_imgs=True)
    gray_pipe.read()
    gray_pipe._square_image()
    gray_pipe._gray_image()
    gray_pipe._resize((128,128))
    gray_pipe.save()

    gray_pipe.vectorize()

    X_user_gray = gray_pipe.features
    y_user_gary = gray_pipe.labels

    '''
    reshape to imgs
    '''

    X_user_color = X_user_color.reshape(X_user_color.shape[0], 128, 128, 3)
    X_user_gray = X_user_gray.reshape(X_user_gray.shape[0], 128, 128, 1)
    
    X_user_color = X_user_color.astype('float32')
    X_user_gray = X_user_gray.astype('float32')
    
    X_user_color = X_user_color/255 # normalizing (scaling from 0 to 1)
    X_user_gray = X_user_gray/255  # normalizing (scaling from 0 to 1)

    '''
    Featurize user img and get cosine dist
    '''

    X_extended_latent = np.hstack((rgb.latent, gray.latent))

    rgb._extract_latent(X_user_color)
    gray._extract_latent(X_user_gray)

    X_user_extended = np.hstack((rgb.latent, gray.latent))

    distances = cosine_similarity(X_user_extended, X_extended_latent)
    # pdb.set_trace()
    X_closest = X_total[np.argsort(distances)][0][::-1][:9]

    X_close_imgs = X_closest.reshape(X_closest.shape[0], 128, 128, 3)

    img_plot_3x3(X_close_imgs, 'closest_to_user')

    '''
    Import df
    '''
    df = pd.read_csv('../data/metadata/2020-06-09_pg1_3_all.csv')



    '''
    Predict on user img with loaded models
    '''
