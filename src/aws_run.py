# general imports
import argparse
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

# other file imports
from cnn_new import Autoencoder
from cnn_w_meta import XceptionAE

# from data_pipelines import ImagePipeline


if __name__ == "__main__":
    # arguments for cmdline control
    parser = argparse.ArgumentParser(description='Run cnn on AWS')
    parser.add_argument('-e','--epochs',help='Set number of Epochs to run')
    parser.add_argument('-b','--batchsize',help='Set batch size for training')
    parser.add_argument('-g','--grayscale', help='Use grayscale imgs, bool, 0 or 1')
    parser.add_argument('-m','--model', help='Name of AE model (XceptionAE or Autoencoder)')
    args = parser.parse_args()

    # grayscale tag
    use_gray = bool(int(args.grayscale))

    # up/down params for exploratory run
    epochs = int(args.epochs)
    batch_1 = int(args.batchsize)       # around 30-40
    kernel_sizes = [(3,3),(4,4),(5,5)]
    layers = 5
    init_filter = 64

    epoch_list = np.array([epochs//2, epochs, epochs*2]).astype(int)
    batch_list = np.array([batch_1-10, batch_1, batch_1+10]).astype(int)

    
    # for _ in range(3):
    if use_gray:
        # Select model
        if args.model == 'XceptionAE':
            model_builder = XceptionAE()
        elif args.model == 'Autoencoder':
            model_builder = Autoencoder()

        # load data from file
        model_builder.load_Xy('2020-06-10')

        X = model_builder.X_total

        X_tt, X_holdout = train_test_split(X, random_state=33)
        X_train, X_test = train_test_split(X_tt, random_state=33)

        X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)
        X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)

    else:
        # Select model
        if args.model == 'XceptionAE':
            model_builder = XceptionAE(gray_imgs=False)
        elif args.model == 'Autoencoder':
            model_builder = Autoencoder(gray_imgs=False)

        # load data from file
        model_builder.load_Xy('2020-06-10')

        X = model_builder.X_total

        X_tt, X_holdout = train_test_split(X, random_state=33)
        X_train, X_test = train_test_split(X_tt, random_state=33)

        X_train = X_train.reshape(X_train.shape[0], 128, 128, 3)
        X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)

    model_builder.build_autoencoder(init_filter, layers, enc_do=0, dec_do=0)

    # fit model
    model_builder.fit_(X_train, X_test, epochs, batch_1)
    
    # save
    model_builder.save_model(args.model)