import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import generatePairs, computeDistanceForPairs
from common.Enums import DistanceMetrics, LossType
from common.computeAccuracy import computeAccuracy
import time
import itertools

DATA_DIR = '/Users/ckanitkar/Desktop/img_npy_feature_only_train_test_subsample/DRESSES/Skirt/'
SHOP_FEAUTRES_DIR = '/Users/ckanitkar/Desktop/img_npy_feature_only/DRESSES/Skirt/'
SHOP_LABELS_DIR = '/Users/ckanitkar/Desktop/labels_only/DRESSES/Skirt/'


consumer_features = np.load(DATA_DIR+ 'train_consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'train_consumer_labels.npy')

shop_features = np.load(SHOP_FEAUTRES_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(SHOP_LABELS_DIR + 'shop_labels.npy')

test_consumer_features = np.load(DATA_DIR + 'test_consumer_ResNet50_features.npy')
test_consumer_labels = np.load(DATA_DIR + 'test_consumer_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

print(test_consumer_features.shape)
print(test_consumer_labels.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine]
lossTypes = [LossType.SVM]
optimizers = ['sgd'] #, 'rmsprop', 'adam']

BATCH_SIZE = 100
EPOCHS = 1
SAVE_MODELS = False


for metric, lossType, optimizer in itertools.product(metrics, lossTypes, optimizers):
	print(metric, lossType, optimizer)

	for epoch in range(EPOCHS):
		print("Starting epoch ", epoch)
		batch_iter = 1
		num_batches = consumer_features.shape[0] // BATCH_SIZE + 1
		for start in range(0, consumer_features.shape[0], BATCH_SIZE):
			last_index = min(consumer_features.shape[0], start + BATCH_SIZE)

			consumer_batch = consumer_features[start: last_index]
			consumer_labels_batch = consumer_labels[start:last_index]

			pair, target, _ = generatePairs(consumer_batch, consumer_labels_batch, shop_features, shop_labels, lossType = lossType, verbose = 1)
			distance = computeDistanceForPairs(pair, metric = metric)

			input_dim = consumer_features.shape[-1]
			hidden_dim = 2048

			model = GetSiameseNet(input_dim,hidden_dim, lossType = lossType, optimizer = optimizer)


			H = model.train_on_batch(distance, target, class_weight = {1: 500, -1: 1})

			print("Finished batch {} of {}".format(batch_iter, num_batches))
			batch_iter += 1





	computeAccuracy(test_consumer_features, shop_features, test_consumer_labels, shop_labels, metric=metric, model=model, k=[10, 20, 30])

	#TODO: Print loss and accuracy

	# Save model

	if(SAVE_MODELS):
		model_json = model.to_json()
		model.save_weights("model_"+ str(metric)+"_"+ lossType.name +"_"+optimizer+"_"+timestr+".h5")

		with open("model_"+ str(metric)+"_"+ lossType.name + "_"+optimizer+"_"+timestr+".json", "w") as json_file:
			json_file.write(model_json)


	






