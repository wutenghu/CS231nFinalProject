import numpy as np
from common.Enums import DistanceMetrics, LossType

def generatePairs(consumer_features, consumer_labels, shop_features, shop_labels, lossType):
	pairs_0 = []
	pairs_1 = []
	targets= [] #np.zeros((N,))
	index_metadata = []

	for j,c in enumerate(consumer_features):
		shop_images_idx = np.where(shop_labels == consumer_labels[j])[0]
		for s in shop_images_idx:
			pairs_0.append(c)
			pairs_1.append(shop_features[s])
			targets.append(1)
			index_metadata.append((j, s))

		shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0]
		assert (len(shop_images_idx) + len(shop_images_idx_neg) == len(shop_labels)), 'Must select all shop images to train against'
		# add neagtive samples
		for s in shop_images_idx_neg:
			pairs_0.append(c)
			pairs_1.append(shop_features[s])
			if (lossType == LossType.BinaryCrossEntropy):
				targets.append(0)
			elif (lossType == LossType.SVM):
				targets.append(-1)
			else:
				raise Exception("Invalid loss type")
			index_metadata.append((j, s))

	return [np.asarray(pairs_0), np.asarray(pairs_1)], np.asarray(targets), index_metadata

def computeDistanceForPairs(data, metric = DistanceMetrics.L1):
	'''
		data (pairs)  ((N,dim), (N,dim))
		data (triplets) ((N,dim), (N,dim), (N,dim))
	'''
	assert isinstance(data, list) and len(data) == 2, 'Data must be a list of pairs. With 1st being consumer, and second being shop'

	consumer = data[0]
	shop = data[1]
	assert consumer.shape == shop.shape
	if metric == DistanceMetrics.L1:
		difference = consumer - shop
		return np.abs(difference)

	elif metric == DistanceMetrics.L2:
		difference = np.power((consumer - shop),2)
		return difference

	elif metric == DistanceMetrics.Cosine:
		difference = 1 - consumer*shop
		return difference

	else:
		raise Exception("Must use a valid distance metric")


