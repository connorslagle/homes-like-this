import os
import shutil
import numpy as np



if __name__=="__main__":
    image_dir = '../data/listing_images/full/all'
    image_fnames = os.listdir(image_dir)

    validation_fnames = np.random.choice(image_fnames, len(image_fnames)//5,replace=False)

    val_dir = '../data/listing_images/full/validation'
    test_dir = '../data/listing_images/full/test'

    for image in validation_fnames:
        source = image_dir + '/' + image
        dest = val_dir + '/' + image
        shutil.move(source, dest)

    test_fnames = os.listdir(image_dir)

    for image in test_fnames:
        source = image_dir + '/' + image
        dest = test_dir + '/' + image
        shutil.move(source, dest)