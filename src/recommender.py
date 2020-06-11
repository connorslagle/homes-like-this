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
from plotting import similar_img_plot, listing_map


if __name__ == "__main__":
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    '''
    Import models and img latents
    '''

    # model names updated on friday
    gray_fname = 'gray_061020_2020-06-10_22:42:13.393938'
    rgb_fname = 'rgb_061020_2020-06-10_22:42:09.415735'

    # latent fnames
    gray_latent_fname = 'gray_061020_2020-06-10_22:42:17.322067_xtest_encode.pkl'
    rgb_latent_fname = 'rgb_061020_2020-06-10_22:42:13.297335_xtest_encode.pkl'

    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    rgb.load_model(rgb_fname, rgb_latent_fname)
    gray.load_model(gray_fname, gray_latent_fname)

    rgb.load_Xy('2020-06-10')
    gray.load_Xy('2020-06-10')

    X_rgb = rgb.X_total
    # X_gray = gray.X_total
    # pdb.set_trace()
    # X_rgb = X_rgb.reshape(X_rgb.shape[0], 128, 128, 3)
    # X_gray = X_gray.reshape(X_gray.shape[0], 128, 128, 1)
    
    # X_rgb = X_rgb.astype('float32')
    # X_gray = X_gray.astype('float32')
    
    # X_rgb = X_rgb/255 # normalizing (scaling from 0 to 1)
    # X_gray = X_gray/255  # normalizing (scaling from 0 to 1)

    # rgb._extract_latent(X_rgb)
    # gray._extract_latent(X_gray)

    # rgb.save_model('rgb_061020')
    # gray.save_model('gray_061020')
    # X_total = np.vstack((X_rgb['train'], X_rgb['test'], X_rgb['holdout']))

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
    X_closest = X_rgb[np.argsort(distances)][0][::-1][:9]

    X_close_imgs = X_closest.reshape(X_closest.shape[0], 128, 128, 3)

    similar_img_plot(X_close_imgs, 2, 'closest_to_user')

    '''
    Import df
    '''
    df = pd.read_csv('../data/metadata/2020-06-09_pg1_3_all.csv')

    hrefs = np.array(rgb.y_href)
    num_unique = 0
    idx = 1
    while num_unique < 10:
        idx += 1
        sorted_href = hrefs[np.argsort(distances)][0][::-1][:idx]
        num_unique = np.unique(sorted_href).shape[0]

    top_ten = sorted_href


    '''
    Top ten with presentation example

    '/realestateandhomes-detail/8091-S-Clayton-Cir_Centennial_CO_80122_M11235-45960', --    7409-S-Dahlia-Ct-Centennial-CO-80122
    '/realestateandhomes-detail/804-S-Vance-St-Unit-B_Lakewood_CO_80226_M17826-33954', -->  11321-W-18th-Ave-Lakewood-CO-80215
    '/realestateandhomes-detail/8560-W-81st-Dr_Arvada_CO_80005_M12329-61732', -->           6189-Iris-Way-Arvada-CO-80004
    '/realestateandhomes-detail/11321-W-18th-Ave_Lakewood_CO_80215_M20461-22788' -->        8060-W-9th-Ave-80214/unit-233
    '/realestateandhomes-detail/1926-Jamaica-St_Aurora_CO_80010_M14147-60408'               17833 E Loyola Ave
    '/realestateandhomes-detail/8843-Flattop-St_Arvada_CO_80007_M24540-74250',              18983-w-95th-ln-arvada-co-80007-
    '/realestateandhomes-detail/8023-Wolff-St-Unit-D_Westminster_CO_80031_M14214-51983',    9690-brentwood-way-broomfield-co
    '/realestateandhomes-detail/8640-W-Alameda-Ave_Lakewood_CO_80226_M24028-57580'          1330 S Youngfield Ct, Lakewood, CO 80228
    '/realestateandhomes-detail/3411-W-98th-Dr-Unit-C_Westminster_CO_80031_M15814-33299',   2901-W-81st-Avenue/Westminster/CO/80031
    '/realestateandhomes-detail/3805-W-84th-Ave_Westminster_CO_80031_M10115-77027']         CO/Arvada/7760-W-87th-Dr-80005/unit-E


    lat = [39.588828, 39.745790, 39.809788, 39.730797, 39.646233, 39.870414, 39.873274, 39.692519, 39.843736, 39.853361]
    long = [-104.931643, -105.125263, -105.106349, -105.087823, -104.780792, -105.215168, -105.093423, -105.142520,-105.023142, -105.085782]
    '''

