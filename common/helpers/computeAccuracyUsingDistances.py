import numpy as np

'''
Computes accuracy of selecting correct shop photo label in k closest shop photos using distance matrix
'''
def computeAccuracyUsingDistances(distances, consumer_labels, shop_labels, k = []):

    assert isinstance(k, list)
    assert isinstance(distances, np.ndarray), 'Distances must be a numpy array of consumer * shop'
    assert isinstance(consumer_labels, np.ndarray), 'Consumer labels must be a numpy array of size (n,)'
    assert isinstance(shop_labels, np.ndarray), 'Shop labels must be a numpy array of size (m,)'
    assert distances.shape == (consumer_labels.shape[0], shop_labels.shape[0]), 'Distances shape must be (# consumer labels by # shop labels)'

    sorted_distances = distances.argsort(axis=1)

    output = []
    for k_val in k:

        closest_k_shop_photos = sorted_distances[:, :k_val]

        closest_k_shop_labels = shop_labels[closest_k_shop_photos]

        closest_k_shop_contains_consumer_label = (closest_k_shop_labels==consumer_labels[:,None]).any(1)

        correct = np.sum(closest_k_shop_contains_consumer_label)
        total = distances.shape[0]
        print ("Correct: {} for k: {}".format(correct, k_val))
        accuracy = correct / total
        print ("Accuracy: {} for k: ".format(accuracy, k_val))
        output.append((correct, total, accuracy))

    assert len(output) == len(k)
    return output