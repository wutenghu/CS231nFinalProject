import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import generatePairs, computeDistanceForPairs
from common.Enums import DistanceMetrics, LossType
from common.computeAccuracy import computeAccuracy
import time
import itertools

DATA_DIR = '/Users/ckanitkar/Desktop/img_npy_feature_only_train_test_subsample/DRESSES/Skirt/'
SHOP_DATA_DIR = '/Users/ckanitkar/Desktop/img_npy_feature_only/DRESSES/Skirt/'
LABELS_DIR = '/Users/ckanitkar/Desktop/labels_only/DRESSES/Skirt/'
FEATURE_TYPE = 'ResNet50'

consumer_features = np.load(DATA_DIR + 'train_consumer_{}_features.npy'.format(FEATURE_TYPE))
consumer_labels = np.load(DATA_DIR + 'train_consumer_labels.npy')

consumer_features = consumer_features[:200, :]
consumer_labels = consumer_labels[:200]
shop_features = np.load(SHOP_DATA_DIR + 'shop_{}_features.npy'.format(FEATURE_TYPE))
shop_labels = np.load(LABELS_DIR + 'shop_labels.npy')

test_consumer_features = np.load(DATA_DIR + 'test_consumer_{}_features.npy'.format(FEATURE_TYPE))
test_consumer_labels = np.load(DATA_DIR + 'test_consumer_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine]
lossTypes = [LossType.BinaryCrossEntropy] #LossType.SVM]
optimizers = ['sgd'] #, 'rmsprop', 'adam']

SAVE_MODEL = False
BATCH_SIZE = 32
EPOCHS = 2
for metric, lossType, optimizer in itertools.product(metrics, lossTypes, optimizers):
	print(metric, lossType, optimizer)

	# Set up model
	input_dim = consumer_features.shape[-1]
	hidden_dim = 2048
	model = GetSiameseNet(input_dim,hidden_dim, lossType = lossType, optimizer = optimizer)

	for epoch in range(EPOCHS):
		print("Epoch", epoch)
		num_batches = consumer_features.shape[0] // BATCH_SIZE + 1
		batch_iter = 1
		for start in range(0, consumer_features.shape[0], BATCH_SIZE):
			last_index = min(consumer_features.shape[0], start + BATCH_SIZE)
			consumer_batch = consumer_features[start: last_index]
			consumer_labels_batch = consumer_labels[start: last_index]

			pair, target, _ = generatePairs(consumer_batch, consumer_labels_batch, shop_features, shop_labels, lossType = lossType)
			distance = computeDistanceForPairs(pair, metric = metric)
			model.fit(distance, target, validation_split=0, epochs=1, class_weight={1: 500, 0: 1}, verbose = 0)


			print("Finished batch {} of {}".format(batch_iter, num_batches))
			batch_iter += 1

	print("Printing train set accuracy")
	computeAccuracy(consumer_features, shop_features, consumer_labels, shop_labels, metric=metric, model=model, k=[10, 20, 30])

	print("Printing test set acuracy")
	computeAccuracy(test_consumer_features, shop_features, test_consumer_labels, shop_labels, metric=metric, model=model,
					k=[10, 20, 30])


	if (SAVE_MODEL):
		model_json = model.to_json()
		model.save_weights("model_" + str(metric) + "_" + lossType.name + "_" + optimizer + "_" + timestr + ".h5")
		with open("model_"+ str(metric)+"_"+ lossType.name + "_"+optimizer+"_"+timestr+".json", "w") as json_file:
			json_file.write(model_json)

	






