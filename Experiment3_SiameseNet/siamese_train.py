import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import generatePairs, computeDistanceForPairs
from common.Enums import DistanceMetrics, LossType
from common.computeAccuracy import computeAccuracy
import time
import itertools

DATA_DIR = './img_npy_feature_only/DRESSES/Skirt/'
LABELS_DIR = './labels_only/DRESSES/Skirt/'

consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(LABELS_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(LABELS_DIR + 'shop_labels.npy')


print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine]
lossTypes = [LossType.BinaryCrossEntropy] #LossType.SVM]
optimizers = ['sgd'] #, 'rmsprop', 'adam']

for metric, lossType, optimizer in itertools.product(metrics, lossTypes, optimizers):
	print(metric, lossType, optimizer)
	consumer_features = consumer_features[:300, :]
	consumer_labels = consumer_labels[:300]

	pair, target, _ = generatePairs(consumer_features, consumer_labels, shop_features, shop_labels, lossType = lossType)
	distance = computeDistanceForPairs(pair, metric = metric)

	input_dim = consumer_features.shape[-1]
	hidden_dim = 2048

	model = GetSiameseNet(input_dim,hidden_dim, lossType = lossType, optimizer = optimizer)

	H = model.fit(distance, target, validation_split=0, epochs = 2, class_weight = {1: 500, 0: 1})
	model_json = model.to_json()
	model.save_weights("model_"+ str(metric)+"_"+ lossType.name +"_"+optimizer+"_"+timestr+".h5")

	computeAccuracy(consumer_features, shop_features, consumer_labels, shop_labels, metric=metric, model=model, k=[10, 20, 30])

	#TODO: Print loss and accuracy

	# Save model

	with open("model_"+ str(metric)+"_"+ lossType.name + "_"+optimizer+"_"+timestr+".json", "w") as json_file:
		json_file.write(model_json)

	






