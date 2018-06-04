import numpy as np
from common.Enums import DistanceMetrics

def LoadData(consumer_features, consumer_labels, shop_features, shop_labels):
	pairs_0 = []
	pairs_1 = []
	targets= [] #np.zeros((N,))
	index_metadata = []

	i = 0
	for j,c in enumerate(consumer_features):
		shop_images_idx = np.where(shop_labels == consumer_labels[j])
		for s in shop_images_idx[0]:
			pairs_0.append(c)
			pairs_1.append(shop_features[s])
			targets.append(1)
			index_metadata.append((j, s))
			i +=1

		shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0][0:10]
		# add neagtive samples
		for s in shop_images_idx_neg:
			#pairs[0][i,:] = c
			pairs_0.append(c)
			pairs_1.append(shop_features[s])
			targets.append(0)
			index_metadata.append((j, s))
			i+=1

	print (np.asarray(pairs_0).shape)
	print (np.asarray(pairs_1).shape)
	return [np.asarray(pairs_0), np.asarray(pairs_1)], np.asarray(targets), index_metadata

def ComputeDistance(data, metric = DistanceMetrics.L1):
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

