import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# lsun
from tensorflow..keras.datasets import 