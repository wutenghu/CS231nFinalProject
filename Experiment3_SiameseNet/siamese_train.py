import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import generatePairs, computeDistanceForPairs
from common.Enums import DistanceMetrics, LossType
from common.computeAccuracy import computeAccuracy
import time
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

DIR_PREFIX = "/Users/ckanitkar/Desktop/"
CLOTHING_TYPE = "DRESSES/Skirt/"
FEATURE_TYPE = 'ResNet50'

DATA_DIR = DIR_PREFIX + 'img_npy_feature_only_train_test_subsample/' + CLOTHING_TYPE
SHOP_DATA_DIR = DIR_PREFIX + 'img_npy_feature_only/' + CLOTHING_TYPE
LABELS_DIR = DIR_PREFIX + 'labels_only/' + CLOTHING_TYPE


POSITIVE_CLASS_WEIGHT = 500
HIDDEN_DIM = 2048


metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine]
lossTypes = [LossType.BinaryCrossEntropy] #LossType.SVM]
optimizers = ['sgd'] #, 'rmsprop', 'adam']

SAVE_MODEL = False
BATCH_SIZE = 32
EPOCHS = 2


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

for metric, lossType, optimizer in itertools.product(metrics, lossTypes, optimizers):
	print(metric, lossType, optimizer)

	# Set up model
	input_dim = consumer_features.shape[-1]
	model = GetSiameseNet(input_dim, HIDDEN_DIM, lossType = lossType, optimizer = optimizer)

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
			negative_key = -1 if(lossType == LossType.SVM) else 0
			model.fit(distance, target, validation_split=0, epochs=1, class_weight={1: POSITIVE_CLASS_WEIGHT, negative_key: 1}, verbose = 1)

			print("Precision Recall on batch")
			preds = model.predict(distance)
			preds = [1 if p > 0.5 else 0 for p in preds]
			print("accuracy on batch:", accuracy_score(target, preds))
			print(confusion_matrix(target, preds))
			print(precision_recall_fscore_support(target, preds, average='weighted'))


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

	






