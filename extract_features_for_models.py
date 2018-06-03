from common.extract_features_functions import *
from Experiment1_WhiteBoxFeatures.feature_extraction_methods import hog_feature
from keras.applications import *
import numpy as np

directory = "/Users/ckanitkar/Desktop/img_npy_final/"


model = VGG19()
layer_name  = ["fc1"]
includedCategories = []


print(model.name)
extract_features_pre_trained(directory, includedCategories = includedCategories, imageReshape = 224, model = model, layer_name=layer_name)


