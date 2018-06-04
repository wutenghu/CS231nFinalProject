import numpy as np
from .Enums import DistanceMetrics
from .computeDistances import computeDistances

'''
Computes accuracy of selecting correct shop photo label in k closest shop photos using distance matrix 
'''
def computeAccuracyUsingDistances(distances, consumer_labels, shop_labels, k = 20):

    assert isinstance(distances, np.ndarray), 'Distances must be a numpy array of consumer * shop'
    assert isinstance(consumer_labels, np.ndarray), 'Consumer labels must be a numpy array of size (n,)'
    assert isinstance(shop_labels, np.ndarray), 'Shop labels must be a numpy array of size (m,)'
    assert distances.shape == (consumer_labels.shape[0], shop_labels.shape[0]), 'Distances shape must be (# consumer labels by # shop labels)'

    closest_k_shop_photos = distances.argsort(axis=1)[:, :k]

    closest_k_shop_labels = shop_labels[closest_k_shop_photos]

    closest_k_shop_contains_consumer_label = (closest_k_shop_labels==consumer_labels[:,None]).any(1)

    correct = np.sum(closest_k_shop_contains_consumer_label)
    total = distances.shape[0]
    print ("Correct: {}".format(correct))
    accuracy = correct / total
    print ("Accuracy: {}".format(accuracy))

    return correct, total, accuracy


def computeAccuracy(consumer_features, shop_features, consumer_labels, shop_labels, k = 20, metric = None, model = None, batchSize = 100):
    assert metric is not None, 'Must specify a metric'

    distances = computeDistances(consumer_features, shop_features, metric, model, batchSize)
    return computeAccuracyUsingDistances(distances, consumer_labels, shop_labels)
