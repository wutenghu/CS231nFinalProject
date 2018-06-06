import numpy as np
import glob

DIRECTORY_PATH = "/Users/ckanitkar/Desktop/img_npy_feature_only_train_test_subsample/"
SUBSAMPLE = 2000

def splitData(dirName):


    consumer_labels = np.load(dirName + "consumer_labels.npy")
    N = consumer_labels.shape[0]
    indices = np.random.permutation(N)[:SUBSAMPLE]
    train_idx, test_idx = indices[:int(0.8 * SUBSAMPLE)], indices[int(0.8 * SUBSAMPLE):]

    assert len(train_idx) + len(test_idx) == SUBSAMPLE, "Missing some indieces"


    # Save Lables

    train_consumer_labels, test_consumer_labels = consumer_labels[train_idx], consumer_labels[test_idx]
    np.save(dirName + "test_consumer_labels.npy", test_consumer_labels)
    np.save(dirName + "train_consumer_labels.npy", train_consumer_labels)

    # Save Features

    features = ["ResNet50", "inception_v3", "vgg16", "vgg19", "whitebox"]
    for feature in features:
        name_tail = "consumer_{}_features.npy".format(feature)
        filename = dirName + name_tail
        consumer_features = np.load(filename)

        assert consumer_features.shape[0] == N, "Features must have same length as labels"

        train_consumer_features, test_consumer_features = consumer_features[train_idx], consumer_features[test_idx]

        np.save(dirName + "train_" + name_tail, train_consumer_features)

        np.save(dirName + "test_" + name_tail, test_consumer_features)





categories = ["DRESSES/Dress", "DRESSES/Skirt", "CLOTHING/UpperBody", "CLOTHING/LowerBody"]

for category in categories:
    path = DIRECTORY_PATH + category + "/"
    splitData(path)
