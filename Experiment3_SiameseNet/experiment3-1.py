import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import LoadData, ComputeDistance

from common.computeDistanceBetweenExtractedFeatures import computeDistances
from common.computeAccuracy import computeAccuracy
from common.Enums import DistanceMetrics

DATA_DIR = './img_npy_features_only/DRESSES/Skirt/'

consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

pair, target = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=True)
distance = ComputeDistance(pair, pairs=True)

input_dim = consumer_features.shape[-1]
hidden_dim = 2048
model = GetSiameseNet(input_dim,hidden_dim)

model.fit(distance, target, validation_split=.2)

feat_distances = computeDistances(consumer_features, shop_features, metric = DistanceMetrics.L1, model=model, batchSize = 100)
print (feat_distances.shape)








