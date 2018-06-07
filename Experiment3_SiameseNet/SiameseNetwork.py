from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop
from keras.losses import binary_crossentropy
from common.Enums import LossType


def GetSiameseNet(input_dim, hidden_dim, lossType, learning_rate, num_hidden_layers = 1, optimizer = 'adam'):
	assert isinstance(learning_rate, float)
	input = Input(shape=(input_dim,))
	output = input
	for i in range(num_hidden_layers):
		output = Dense(hidden_dim, activation='relu')(output)

	if optimizer == 'rmsprop':
		optimizer = RMSprop(lr = learning_rate)
	elif optimizer == 'sgd':
		# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr = learning_rate)

	# sigmoid loss
	if lossType == LossType.BinaryCrossEntropy:
		output = Dense(1, activation='sigmoid')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss='binary_crossentropy')

	# contrastive loss
	elif lossType == LossType.SVM:
		output = Dense(1, activation='linear')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss='hinge')

	else:
		raise Exception("Must provide a loss type")
	return siamese_net