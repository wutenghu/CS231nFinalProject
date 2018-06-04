import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import generatePairs, computeDistanceForPairs
from common.Enums import DistanceMetrics, LossType
import time

DATA_DIR = '/Users/ckanitkar/Desktop/img_npy_final_features_only/DRESSES/Skirt/'


consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

#TODO: split into train val test and save npy files

timestr = time.strftime("%Y%m%d-%H%M%S")

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine]
lossTypes = [LossType.SVM]
optimizers = ['sgd'] #, 'rmsprop', 'adam']

for metric, lossType in zip(metrics, lossTypes):
	pair, target, _ = generatePairs(consumer_features, consumer_labels, shop_features, shop_labels, lossType = lossType)
	distance = computeDistanceForPairs(pair, metric = metric)

	input_dim = consumer_features.shape[-1]
	hidden_dim = 2048

	final_activation = ['sigmoid'] #, 'svm']

	for optimizer in optimizers:
		print ("optimizer:", optimizer)
		model = GetSiameseNet(input_dim,hidden_dim, lossType = lossType, optimizer = optimizer)

		#batch_size=32
		H = model.fit(distance, target, validation_split=.2, epochs = 10, class_weight = {1: 500, 0: 1})
		model_json = model.to_json()
		model.save_weights("model_"+ str(metric)+"_"+ lossType.name +"_"+optimizer+"_"+timestr+".h5")

		#TODO: Print loss and accuracy

		# Save model

		with open("model_"+ str(metric)+"_"+ lossType.name + "_"+optimizer+"_"+timestr+".json", "w") as json_file:
			json_file.write(model_json)

	






