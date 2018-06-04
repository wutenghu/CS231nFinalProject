from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop
from keras.losses import binary_crossentropy


def GetSiameseNet(input_dim, hidden_dim, final_activation = 'sigmoid', optimizer = 'adam'):
	input = Input(shape=(input_dim,))
	output = Dense(hidden_dim, activation='relu')(input)

	if optimizer == 'rmsprop':
		optimizer = RMSprop()
	elif optimizer == 'sgd':
		optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr = 0.001)

	# sigmoid loss
	if final_activation == 'sigmoid':
		output = Dense(1, activation='sigmoid')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss='binary_crossentropy')

	# contrastive loss
	elif final_activation == 'svm':
		output = Dense(1, activation='linear')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer=optimizer, loss='binary_hinge')

	return siamese_net