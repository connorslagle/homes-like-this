# general imports
import argparse
import pickle


# other file imports
from cnn_new import Autoencoder
from data_pipelines import ImagePipeline


if __name__ == "__main__":
    # arguments for cmdline control
    parser = argparse.ArgumentParser(description='Run cnn on AWS')
    parser.add_argument('-e','--epochs',help='Number of Epochs to run model')
    parser.add_argument('-b','--batch-size',help='Set batch size for training')
    args = parser.parse_args()

    # preprocess gray/color imgs
    # color
    color = ImagePipeline('../data/proc_imgs/128/color')
    gray = ImagePipeline('../data/proc_imgs/128/gray')

    

