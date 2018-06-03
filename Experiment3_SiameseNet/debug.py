from keras.models import model_from_json
from common.computeDistances import computeDistances, computeFeatureWiseMetric
from common.computeAccuracy import computeAccuracy
from common.DistanceMetrics import DistanceMetrics, LossType
from SiameseDataUtil import ComputeDistance, LoadData
import numpy as np
import matplotlib.pyplot as plt



MODEL_PATH = './model_DistanceMetrics.L1_sigmoid_sgd_20180602-182415.json'
WEIGHTS_PATH = './model_DistanceMetrics.L1_sigmoid_sgd_20180602-182415.h5'

json_file = open(MODEL_PATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(WEIGHTS_PATH)

DATA_DIR = '/Users/ckanitkar/Desktop/img_npy_final/DRESSES/Skirt/'

consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')
consumer_photos = np.load(DATA_DIR + 'consumer_photos.npy')
shop_photos = np.load(DATA_DIR + 'shop_photos.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)
print()


pair, target, index_metadata, metadata_by_consumer_image = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=True, lossType = LossType.SVM)

#distances = ComputeDistance(pair, True, DistanceMetrics.L1)

consumer_image_index  = 0
same_as_consumer_index_s = metadata_by_consumer_image[consumer_image_index]['same']
not_same_as_consumer_index_s = metadata_by_consumer_image[consumer_image_index]['different']

print(metadata_by_consumer_image[0])


# correct_indexes = np.where(target == 1)[0]
# incorrect_indexes = np.where(target == 0)[0]
#
# print(target.shape)
#
#
#
# correct_tuples = index_metadata[correct_indexes]
# print(correct_tuples.shape)
# incorrect_tuples = index_metadata[incorrect_indexes]
#
# same_index = 4
# not_same_index = 4
#
# tuple = correct_tuples[same_index]





#distances_correct = distances[correct_indexes][same_index]
#distances_incorrect = distances[incorrect_indexes][not_same_index]
#distances2 = computeFeatureWiseMetric(consumer_features, shop_features, metric=DistanceMetrics.L1)


# prediction = model.predict(distances_correct.reshape((1, -1)))
# print(prediction)
# prediction = model.predict(distances_incorrect.reshape((1, -1)))
# print(prediction)



# Test 2



for consumer_image_index in range(4):
    print("Consumer image index: {}".format(consumer_image_index))

    for same_as_consumer_index in same_as_consumer_index_s:
        print("Correct shop index: {}".format(same_as_consumer_index))
        distance_same = np.abs(consumer_features[consumer_image_index] - shop_features[same_as_consumer_index]).reshape(
(1, -1))
        print(model.predict(distance_same))

    for not_same_as_consumer_index in not_same_as_consumer_index_s:
        print(" In Correct shop index: {}".format(not_same_as_consumer_index))
        distance_different = np.abs(consumer_features[consumer_image_index] - shop_features[not_same_as_consumer_index]).reshape((1, -1))
        print(model.predict(distance_different))


# total_distances = computeDistances(consumer_features, shop_features, model = model)
#
# print(total_distances[consumer_image_index][same_as_consumer_index])
# print(total_distances[consumer_image_index][not_same_as_consumer_index])
#
# closest_for_selected_consumer = total_distances[consumer_image_index].argsort()
# print (total_distances[consumer_image_index].sort())
#
# correct, total, accuracy = computeAccuracy(total_distances, consumer_labels, shop_labels)
#
#
# #distances2 = distances2[tuple[0], tuple[1], :]
# #print(distance_temp.shape)
#
# #np.array_equal(distances_temp, distances2.squeeze())
# #print(distances_temp)
# #print(distances2.squeeze())
#
#
# # plt.imshow(consumer_photos[tuple[0]].transpose([1,2,0]))
# # plt.show()
# #
# # plt.imshow(shop_photos[tuple[1]].transpose([1,2,0]))
# # plt.show()
#
#
# plt.imshow(consumer_photos[consumer_image_index].transpose([1,2,0]))
# plt.show()
#
# plt.imshow(shop_photos[same_as_consumer_index].transpose([1,2,0]))
# plt.show()
#
# plt.imshow(shop_photos[not_same_as_consumer_index].transpose([1,2,0]))
# plt.show()
#
# for closest in closest_for_selected_consumer[:5]:
#     print(total_distances[consumer_image_index][closest])
#     plt.imshow(shop_photos[closest].transpose([1,2,0]))
#     plt.show()


