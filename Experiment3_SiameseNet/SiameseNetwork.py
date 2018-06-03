from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop
from keras.losses import binary_crossentropy
from common.DistanceMetrics import LossType


from keras import backend as K
def weighted_binary_crossentropy(y_true, y_pred, weight=500 ) :
    logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
    return K.mean( logloss, axis=-1)

def weightedLossGenerator(lossType, weights = None):
	if (lossType == LossType.SVM):
		def weightedHinge(y_true, y_pred):
			return K.mean(K.maximum(1. - y_true * y_pred, 0.) * (1 +  (1 + y_true) * .5 * 500), axis=-1)
		return weightedHinge
	else:
		raise Exception("Need loss type to generate weighted loss function")



def GetSiameseNet(input_dim, hidden_dim, lossType = None, optimizer = 'adam'):
	input = Input(shape=(input_dim,))
	output = Dense(hidden_dim, activation='relu')(input)

	if optimizer == 'rmsprop':
		optimizer = RMSprop()
	elif optimizer == 'sgd':
		optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr = 0.001)

	# sigmoid loss
	if lossType == LossType.BinaryCrossEntropy:
		output = Dense(1, activation='sigmoid')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['binary_accuracy'])

	# contrastive loss
	elif lossType == LossType.SVM:
		output = Dense(1, activation='linear')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss=weightedLossGenerator(lossType))

	else:
		raise Exception("Must provide loss type for final layer.")

	return siamese_net