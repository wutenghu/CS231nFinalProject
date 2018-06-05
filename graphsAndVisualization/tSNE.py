import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import itertools

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
    #              color=plt.cm.Set1(y[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) <12e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(photos[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)






photos = np.load("/Users/ckanitkar/Desktop/photos_rgb_only/CLOTHING/LowerBody/consumer_photos.npy").transpose(
    [0, 2, 3, 1])
#print(X.shape)
#print(photos.shape)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")

#t0 = time()
# dirNames = ["DRESSES/Dress"]
# clothingtypes = ["consumer","shop"]
# for dirName, clothing in itertools.product(dirNames, clothingtypes):
#     prefix = "/Users/ckanitkar/Desktop/img_npy_feature_only/"
#     loadname = prefix + dirName + "/" + clothing + "_ResNet50_features.npy"
#     print(loadname)
#     X = np.load(loadname)
#     print(X.shape)
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, verbose=2)
#     X_tsne = tsne.fit_transform(X)
#     savename = "{}_embedding_{}".format(clothing, dirName.split("/")[-1])
#     print(savename)
#     np.save(savename, X_tsne)

#np.save("embedding_shop", X_tsne)

# ONLY USE CONSUMER EMBEDDINGS. THEY ARE MUCH BETTER
X_tsne = np.load("consumer_embedding_LowerBody.npy")
print(X_tsne.shape)

assert photos.shape[0] == X_tsne.shape[0]

plot_embedding(X_tsne)
plt.show()
