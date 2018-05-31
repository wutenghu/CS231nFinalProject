import numpy as np
import scipy.spatial.distance as scidist
from keras.models import Model
from .DistanceMetrics import DistanceMetrics


def computeFeatureWiseMetric(consumer_batch, shop_features, metric):
    assert consumer_batch.shape[1] == shop_features.shape[1], "Consumer batch and shop features must have same feature dimensin"
    consumer_count = consumer_batch.shape[0]
    shop_count = shop_features.shape[0]
    consumer_batch = np.expand_dims(consumer_batch, axis=1)
    consumer_batch = np.tile(consumer_batch, (1, shop_count, 1))
    shop_features = np.expand_dims(shop_features, axis = 0)
    shop_features = np.tile(shop_features, (consumer_count, 1, 1))

    diff = consumer_batch - shop_features

    if metric == DistanceMetrics.L1:
        return np.abs(diff)
    elif metric == DistanceMetrics.L2:
        return np.square(diff)
    else:
        raise Exception("Invalid metric")

def computeDistances(consumer_features, shop_features, metric=DistanceMetrics.L1, model = None, batchSize = 100):
    assert isinstance(consumer_features, np.ndarray), 'Consumer features must be an numpy array of size n * d'
    assert isinstance(shop_features, np.ndarray), 'Shop features must be a numpy array of size m * d'
    assert consumer_features.shape[1] == shop_features.shape[1], 'Consumer and shop features must have same dimension'

    if model is not None:
      assert isinstance(model, Model), "model must be a keras model"
      result = np.array([]).reshape((-1, shop_features.shape[0]))
      num_batches = consumer_features.shape[0] // batchSize + 1
      batch_iter = 1
      for start in range(0, consumer_features.shape[0], batchSize):
          last_index = min(consumer_features.shape[0], start + batchSize)

          consumer_batch = consumer_features[start: last_index]
          feature_wise_metric = computeFeatureWiseMetric(consumer_batch, shop_features, metric)
          feature_wise_metric = feature_wise_metric.reshape((-1, feature_wise_metric.shape[2]))

          similarity = model.predict(feature_wise_metric)
          # We multiply by negative 1 since higher scores means they are more similar, aka negative of distance.
          similarity = -1 * similarity.reshape((consumer_batch.shape[0], -1))

          print("Finished batch {} of {}".format(batch_iter, num_batches))
          batch_iter +=1
          result = np.concatenate((result, similarity))

      return result

    else:
      if(metric == DistanceMetrics.L1):
          metric_string = 'cityblock'
      elif(metric == DistanceMetrics.L2):
          metric_string = 'euclidean'
      return scidist.cdist(consumer_features, shop_features, metric=metric_string)