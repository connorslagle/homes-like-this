# general imports
import numpy as np

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# other python files
from data_pipelines import ImagePipeline
from cnn_new import Autoencoder
from plotting import pca_var_all, elbow_plot



if __name__ == "__main__":
    # import data
    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    rgb.load_Xy('2020-06-04')
    gray.load_Xy('2020-06-04')

    X_rgb = rgb.X_rgb
    X_gray = gray.X_gray

    # center data
    scaler = StandardScaler()
    X_rbg_ss = scaler.fit_transform(X_rgb['train'])
    X_gray_ss = scaler.fit_transform(X_gray['train'])


    # init pca
    pca_rbg = PCA(n_components=300, random_state=33)
    pca_rbg.fit(X_rbg_ss)

    pca_gray = PCA(n_components=300, random_state=33)
    pca_gray.fit(X_gray_ss)

    # pca_var_all(pca_rbg.explained_variance_ratio_, pca_gray.explained_variance_ratio_)

    # cluster with kmeans, look at top 9 ims
    elbow_plot(pca_rbg.components_, 20, 'color')
    elbow_plot(pca_gray.components_, 20, 'gray')
