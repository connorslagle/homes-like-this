#general imports
import numpy as np
import pandas as pd
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
from data_pipelines import ImagePipeline
from cnn_new import Autoencoder

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
    Import df
    '''
    # df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')



    # '''
    # Load pkled Xy
    # '''


    '''
    Predict on user img with loaded models
    '''
