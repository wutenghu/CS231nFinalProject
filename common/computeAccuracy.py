import numpy as np

from common.helpers.computeDistances import computeDistances
from common.helpers.computeAccuracyUsingDistances import computeAccuracyUsingDistances


'''
Computes Accuracy on consumer features and shop features using a trained model or vanilla distance metric
'''

def computeAccuracy(consumer_features, shop_features, consumer_labels, shop_labels, k = [20], metric = None, model = None, batchSize = 100):
    assert metric is not None, 'Must specify a metric'
    assert consumer_labels.shape == (consumer_features.shape[0], ), "Incorrect consumer labels input"

    distances = computeDistances(consumer_features, shop_features, metric, model, batchSize)
    return computeAccuracyUsingDistances(distances, consumer_labels, shop_labels, k)
