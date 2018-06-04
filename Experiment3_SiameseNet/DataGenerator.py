import numpy as np
import keras
from Experiment3_SiameseNet.SiameseDataUtil import ComputeDistance

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