import numpy as np
from common.extract_features import extract_features_pre_trained
from keras.applications.vgg16 import VGG16, preprocess_input

DATA_DIR = './img_npy'

base_model = VGG16(weights='imagenet', include_top=True)
extract_features_pre_trained(DATA_DIR, base_model, layer_index=-1)