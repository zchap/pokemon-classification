import lasagne
import numpy
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import classification

xTestImages = numpy.array(classification.load_pokemon_images('TrainingImages'))
xTestImages = xTestImages.reshape(-1,3,96,96)


# convolutionalNeuralNet = NeuralNet(
#     layers=[('input', layers.InputLayer),
#             ('conv2d1', layers.Conv2DLayer),
#             ('maxpool1', layers.MaxPool2DLayer),
#             ('conv2d2', layers.Conv2DLayer),
#             ('maxpool2', layers.MaxPool2DLayer),
#             ('dropout1', layers.DropoutLayer),
#             ('dense', layers.DenseLayer),
#             ('dropout2', layers.DropoutLayer),
#             ('output', layers.DenseLayer),
#             ],
#     # input layer
#     input_shape=(None, 1, 28, 28),
#     # layer conv2d1
#     conv2d1_num_filters=32,
#     conv2d1_filter_size=(5, 5),
#     conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
#     conv2d1_W=lasagne.init.GlorotUniform(),
#     # layer maxpool1
#     maxpool1_pool_size=(2, 2),
#     # layer conv2d2
#     conv2d2_num_filters=32,
#     conv2d2_filter_size=(5, 5),
#     conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
#     # layer maxpool2
#     maxpool2_pool_size=(2, 2),
#     # dropout1
#     dropout1_p=0.5,
#     # dense
#     dense_num_units=256,
#     dense_nonlinearity=lasagne.nonlinearities.rectify,
#     # dropout2
#     dropout2_p=0.5,
#     # output
#     output_nonlinearity=lasagne.nonlinearities.softmax,
#     output_num_units=10,
#     # optimization method params
#     update=nesterov_momentum,
#     update_learning_rate=0.01,
#     update_momentum=0.9,
#     max_epochs=10,
#     verbose=1,
#     )
# # Train the network
# nn = convolutionalNeuralNet.fit(X_train, y_train)
#
