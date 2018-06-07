import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import itertools
from Image_File_IO.extract_features_iterator import resizeImage

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
            if np.min(dist) <1e-3:
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
dirNames = ["DRESSES/Dress", "DRESSES/Skirt", "CLOTHING/UpperBody", "CLOTHING/LowerBody"]
clothingtypes = ["consumer"]


# concat = np.array([]).reshape((0, 2048))
# concat_photos = np.array([], dtype=np.uint8).reshape((0, 3,128, 128))
# print(concat_photos.dtype)
# SUBSAMPLE = 400
# for dirName, clothing in itertools.product(dirNames, clothingtypes):
#     prefix = "/Users/ckanitkar/Desktop/img_npy_feature_only/"
#     loadname = prefix + dirName + "/" + clothing + "_ResNet50_features.npy"
#     print(loadname)
#     X = np.load(loadname)
#     photo_prefix = "/Users/ckanitkar/Desktop/photos_rgb_only/"
#     photo_name = photo_prefix + dirName + "/" + clothing + "_photos.npy"
#     print(photo_name)
#     photos = np.load(photo_name)
#     print(photos.dtype)
#     assert (photos.shape[0] == X.shape[0])
#     sample = np.random.choice(X.shape[0], SUBSAMPLE)
#     X_sample = X[sample, :]
#     photo_sample = photos[sample, :, :, :]
#
#     # print(X_sample.shape)
#     # print(photo_sample.shape)
#     concat = np.concatenate((concat, X_sample))
#     concat_photos = np.concatenate((concat_photos, photo_sample))
#     # print(concat.shape)
#     # print(concat_photos.shape)
#
#
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, verbose=2)
# X_tsne = tsne.fit_transform(concat)
# np.save("cross_category_embedding", X_tsne)
# #
# np.save("cross_category_photos", concat_photos)

#ONLY USE CONSUMER EMBEDDINGS. THEY ARE MUCH BETTER
X_tsne = np.load("cross_category_embedding.npy")
photos = np.load("cross_category_photos.npy").transpose([0, 2,3,1])
photos = resizeImage(photos, 32)
print(photos.shape)
print(X_tsne.shape)

assert photos.shape[0] == X_tsne.shape[0]

plot_embedding(X_tsne)
plt.show()
