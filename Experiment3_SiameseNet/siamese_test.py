from keras.models import model_from_json
from common.computeDistances import computeDistances
from common.computeAccuracy import computeAccuracy
from common.Enums import DistanceMetrics
import numpy as np


MODEL_PATH = './model_DistanceMetrics.L1_sigmoid_sgd_20180602-143655.json'
WEIGHTS_PATH = './model_DistanceMetrics.L1_sigmoid_sgd_20180602-143655.h5'

json_file = open(MODEL_PATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(WEIGHTS_PATH)
print("Loaded model from disk")

DATA_DIR = './img_npy_final_features_only/DRESSES/Skirt/'

consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

metrics = [DistanceMetrics.L1] #, DistanceMetrics.L2, DistanceMetrics.Cosine] 
top_k = [3,10,20,30,40,50]

for metric in metrics:
		print ("metric:", metric)
		feat_distances = computeDistances(consumer_features, shop_features, metric = metric, model=model, batchSize = 100)
		print (feat_distances.shape)

		for k in top_k:
			correct, total, accuracy = computeAccuracy(consumer_features,
													   shop_features,
													   consumer_labels,
													   shop_labels,
													   metric = metric,
													   model = model,
													   k = k)
			print ("Top" + str(k) + "accuracy:", accuracy)
