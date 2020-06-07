# general imports
import argparse
import pickle
import numpy as np

# other file imports
from cnn_new import Autoencoder
from data_pipelines import ImagePipeline


if __name__ == "__main__":
    # arguments for cmdline control
    parser = argparse.ArgumentParser(description='Run cnn on AWS')
    parser.add_argument('-e','--epochs',help='Set number of Epochs to run')
    parser.add_argument('-b','--batchsize',help='Set batch size for training')
    parser.add_argument('-g','--grayscale', help='Use grayscale imgs, bool, 0 or 1')
    args = parser.parse_args()

    # grayscale tag
    use_gray = bool(int(args.grayscale))

    # up/down params for exploratory run
    epochs = int(args.epochs)
    batch = int(args.batchsize)
    layers = 5
    init_filter = 64

    epoch_list = np.array([epochs//2, epochs, epochs*2]).astype(int)
    batch_list = np.array([batch/2, batch, batch*2]).astype(int)

    for elem in batch_list:
        if use_gray:
            # load data
            pipeline = ImagePipeline('../data/proc_imgs/128/gray')
            pipeline.build_Xy(set_seed=True)
            X_train, X_test = pipeline.X_train, pipeline.X_test

            # build model
            model = Autoencoder()
            model.build_autoencoder(init_filter, layers)
        else:
            # load data
            pipeline = ImagePipeline('../data/proc_imgs/128/color', gray_imgs=False)
            pipeline.build_Xy(set_seed=True)
            X_train, X_test = pipeline.X_train, pipeline.X_test

            # build model
            model = Autoencoder(gray_imgs=False)
            model.build_autoencoder(init_filter, layers)

        # fit model
        model.fit_(X_train, X_test, epochs, elem)
        
        # save
        model.save_model()