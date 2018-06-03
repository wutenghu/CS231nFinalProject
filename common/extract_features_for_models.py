from common.extract_features_functions import *
from Experiment1_WhiteBoxFeatures.feature_extraction_methods import hog_feature
from keras.applications import *
import numpy as np
from common.DistanceMetrics import LossType

directory = "/Users/ckanitkar/Desktop/img_npy_final/"

#extract_features_white_box(directory, includedCategories=["Skirt"], extractor_functions=[hog_feature])

models = [VGG16(), VGG19(), DenseNet201()]
layers  = ["fc1", "fc1", "avg_pool"]

for model, layer_name in zip(models, layers):
    print(model.name)
    extract_features_pre_trained(directory, imageReshape = 224, model = model, layer_name=layer_name)


