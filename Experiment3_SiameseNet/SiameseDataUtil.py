import numpy as np
from common.Enums import DistanceMetrics

def LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=None, triplets=None):
	if pairs == True:
		pairs_0 = []
		pairs_1 = []
		targets= [] #np.zeros((N,))
		index_metadata = []

		i = 0
		for j,c in enumerate(consumer_features):
			# print ("i:", i)
			# print ("j:", j)
			shop_images_idx = np.where(shop_labels == consumer_labels[j])
			for s in shop_images_idx[0]:
				# print ("s:", s)
				#pairs[0][i,:] = c
				pairs_0.append(c)
				#print (pairs[1].shape)
				# print (shop_features[s].shape)
				#pairs[1][i,:] = shop_features[s]
				pairs_1.append(shop_features[s])
				targets.append(1)
				index_metadata.append((j, s))
				i +=1

			shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0][0:10]
			# print ("shop_images_idx_neg", list(shop_images_idx_neg))
			# add neagtive samples
			for s in shop_images_idx_neg:
				#pairs[0][i,:] = c
				pairs_0.append(c)
				#pairs[1][i,:] = shop_features[s]
				pairs_1.append(shop_features[s])
				targets.append(0)
				index_metadata.append((j, s))
				i+=1

		print (np.asarray(pairs_0).shape)
		print (np.asarray(pairs_1).shape)
		return [np.asarray(pairs_0), np.asarray(pairs_1)], np.asarray(targets), index_metadata

	if triplets == True:
		triplets_0 = []
		triplets_1 = []
		triplets_2 = []

		for j,c in consumer_features:
			shop_images_idx = np.where(shop_labels == consumer_labels[j])
			shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0]
			for s in shop_images_idx[0]:
				triplets_0.append(c)
				triplets_1.append(shop_features[s])
				triplets_2.append(shop_features[np.random.choice(shop_images_idx_neg)])

		print (np.asarray(triplets_0.shape))
		print (np.asarray(triplets_1.shape))
		print (np.asarray(triplets_2.shape))
		return [np.asarray(triplets_0), np.asarray(triplets_1), np.asarray(triplets_2)]

def ComputeDistance(data, pairs=None, triplets=None, metric = DistanceMetrics.L1):
	'''
		data (pairs)  ((N,dim), (N,dim))
		data (triplets) ((N,dim), (N,dim), (N,dim))
	'''

	if pairs == True:
		consumer = data[0]
		shop = data[1]
		assert consumer.shape == shop.shape
		if metric == DistanceMetrics.L1:
			difference = consumer - shop
			return np.abs(difference)

		elif metric == DistanceMetrics.L2:
			difference = np.sqrt(np.power((consumer - shop),2))
			return difference

		elif metric == DistanceMetrics.Cosine:
			difference = 1 - consumer*shop
			return difference

	if triplets==True:
		consumer = data[0]
		shop_pos = data[1]
		shop_neg = data[2]

		if metric == DistanceMetrics.L1:
			distance_positive = consumer - shop_pos
			distance_negative = consumer - shop_neg
			return np.abs(distance_positive - distance_negative)

		elif metric == DistanceMetrics.L2:
			distance_positive = np.power((consumer - shop_pos),2)
			distance_negative = np.power((consumer - shop_neg),2)
			return distance_positive - distance_negative

		elif metric == DistanceMetrics.Cosine:
			distance_positive = 1 - consumer*shop_pos
			distance_negative = 1 - consumer*shop_neg
			return distance_positive - distance_negative

