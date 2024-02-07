import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from IPython.display import clear_output
from matplotlib.pyplot import imshow
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.metrics import *
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as k
import datetime
import os
import csv
import pandas as pd
import random
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, BatchNormalization , Input ,concatenate , Concatenate
from keras.layers import Dense,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, UpSampling2D, Lambda, Multiply
from keras.losses import categorical_crossentropy,categorical_hinge,hinge,squared_hinge
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator 