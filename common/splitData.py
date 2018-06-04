import numpy as np


def splitData(data_dir):
    consumer_features = np.load(data_dir + "consumer_ResNet50_features.npy")
    consumer_labels = np.load(data_dir + "consumer_labels.npy")

    N = consumer_features.shape[0]
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:int(0.7 * N)], indices[int(0.7 * N):]
    print(len(train_idx))

    print(len(test_idx))

    train_consumer_features, test_consumer_features = consumer_features[train_idx], consumer_features[test_idx]
    train_consumer_labels, test_consumer_labels = consumer_labels[train_idx], consumer_labels[test_idx]

    np.save(data_dir + "train_consumer_ResNet50_features.npy", train_consumer_features)
    np.save(data_dir + "train_consumer_labels.npy", train_consumer_labels)
    np.save(data_dir + "test_consumer_ResNet50_features.npy", test_consumer_features)
    np.save(data_dir + "test_consumer_labels.npy", test_consumer_labels)


splitData('./img_npy_final_features_only/CLOTHING/UpperBody/')