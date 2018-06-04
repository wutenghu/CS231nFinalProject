import numpy as np
from common.DistanceMetrics import DistanceMetrics

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pairs, labels, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.pairs = pairs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        pairs_temp = [self.pairs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(pairs_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, pairs_temp, labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        X = []
        y = []
        # Generate data
        for i, pair in enumerate(pairs_temp):
            # Store sample
            X[i,] = ComputeDistance(pair, pairs = True)

            # Store class
            y[i] = self.labels[i]

        return np.asarray(X), np.asarray(y) 


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

			shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0]#[0:10]
			print ("shop_images_idx_neg", len(shop_images_idx_neg))
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
		targets= [] 

		for j,c in enumerate(consumer_features):
			shop_images_idx = np.where(shop_labels == consumer_labels[j])
			shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0]
			for s in shop_images_idx[0]:
				triplets_0.append(c)
				triplets_1.append(shop_features[s])
				triplets_2.append(shop_features[np.random.choice(shop_images_idx_neg)])
				targets.append(0)

		print (np.asarray(triplets_0).shape)
		print (np.asarray(triplets_1).shape)
		print (np.asarray(triplets_2).shape)
		return [np.asarray(triplets_0), np.asarray(triplets_1), np.asarray(triplets_2)], np.asarray(targets)

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
			difference = np.power((consumer - shop),2)
			return difference

		elif metric == DistanceMetrics.Cosine:
			difference = 1 - consumer*shop
			return difference	

	if triplets==True:
		consumer = data[0]
		shop_pos = data[1]
		shop_neg = data[2]

		if metric == DistanceMetrics.L1:
			distance_positive = np.abs(consumer - shop_pos)
			distance_negative = np.abs(consumer - shop_neg)
			return distance_positive - distance_negative

		elif metric == DistanceMetrics.L2:
			distance_positive = np.power((consumer - shop_pos),2)
			distance_negative = np.power((consumer - shop_neg),2)
			return distance_positive - distance_negative

		elif metric == DistanceMetrics.Cosine:
			distance_positive = 1 - consumer*shop_pos
			distance_negative = 1 - consumer*shop_neg
			return distance_positive - distance_negative

