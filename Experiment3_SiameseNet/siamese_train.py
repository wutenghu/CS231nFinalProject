import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import LoadData, ComputeDistance, DataGenerator
from common.DistanceMetrics import DistanceMetrics
import time

DATA_DIR = './img_npy_final_features_only/DRESSES/Skirt/'
PAIRS = True

consumer_features = np.load(DATA_DIR + 'train_consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'train_consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

#TODO: split into train val test and save npy files

timestr = time.strftime("%Y%m%d-%H%M%S")

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine] 
optimizers = ['sgd'] #, 'rmsprop', 'adam']


for metric in metrics:

	if PAIRS:
		data, target, metadata = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=True)
		distance = ComputeDistance(data, pairs=True, metric = metric)

	else:
		data, target = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, triplets=True)
		distance = ComputeDistance(data, triplets=True, metric = metric)

	input_dim = consumer_features.shape[-1]
	hidden_dim = 2048

	final_activation = ['sigmoid'] #, 'svm']

	for activation in final_activation:
		print ("activation:", activation)
		for optimizer in optimizers:
			print ("optimizer:", optimizer)
			model = GetSiameseNet(input_dim,hidden_dim, final_activation = 'sigmoid', optimizer = optimizer)

			#batch_size=32
			H = model.fit(distance, target, validation_split=.2, epochs = 3, class_weight = {1:500, 0:1})
			model_json = model.to_json()
			model.save_weights("model_weighted"+ str(metric)+"_"+activation+"_"+optimizer+"_"+timestr+".h5")

			#TODO: Print loss and accuracy

			if PAIRS:
				with open("model_weighted"+ str(metric)+"_"+activation+"_"+optimizer+"_"+timestr+".json", "w") as json_file:
					json_file.write(model_json)
			else:
				with open("model_triplet_"+ str(metric)+"_" +activation+"_"+optimizer+"_"+timestr+".json", "w") as json_file:
					json_file.write(model_json)

	






