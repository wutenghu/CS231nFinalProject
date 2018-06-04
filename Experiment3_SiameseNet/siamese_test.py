from keras.models import model_from_json
from common.computeDistances import computeDistances
from common.computeAccuracy import computeAccuracy
from common.DistanceMetrics import DistanceMetrics
import numpy as np
from SiameseDataUtil import LoadData, ComputeDistance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# MODEL_PATH = './model_triplet_DistanceMetrics.L2_sigmoid_sgd_20180603-111247.json'
# WEIGHTS_PATH = './model_DistanceMetrics.L2_sigmoid_sgd_20180603-111247.h5'
PAIRS = True

MODEL_PATH = './model_500DistanceMetrics.L1_sigmoid_sgd_20180603-160333.json'
WEIGHTS_PATH = './model_500DistanceMetrics.L1_sigmoid_sgd_20180603-160333.h5'

# MODEL_PATH = './model_500_weightedDistanceMetrics.L1_sigmoid_sgd_20180603-170051.json'
# WEIGHTS_PATH = './model_500_weightedDistanceMetrics.L1_sigmoid_sgd_20180603-170051.h5'

json_file = open(MODEL_PATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(WEIGHTS_PATH)
print("Loaded model from disk")

DATA_DIR = './img_npy_final_features_only/DRESSES/Skirt/'

consumer_features = np.load(DATA_DIR + 'test_consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'test_consumer_labels.npy')
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
	if PAIRS:
		data, target, index_metadata = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=True)
		distance = ComputeDistance(data, pairs=True, metric = metric)
	else:
		data, target = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, triplets=True)
		distance = ComputeDistance(data, triplets=True, metric = metric)

	#train accuracy
	preds = model.predict(distance)
	preds = [1 if p > 0.5 else 0 for p in preds ]
	print (preds[0:30])
	print (target[0:30])
	print ("accuracy:", accuracy_score(target,preds)) 
	print (confusion_matrix(target, preds))
	print (precision_recall_fscore_support(target, preds, average='weighted'))
	
	feat_distances = computeDistances(consumer_features, shop_features, metric = metric, model=model, batchSize = 100)
	# print (feat_distances.shape)

	for k in top_k:
		correct, total, accuracy = computeAccuracy(feat_distances, consumer_labels, shop_labels, k = k)
		print ("Top" + str(k) + "accuracy:", accuracy)

	
