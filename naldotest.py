import numpy
import classification

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

xTestImages = numpy.array(classification.load_pokemon_images('TrainingImages'))
xTestImages = xTestImages.reshape(-1,3,96,96)
y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')

layersOfCNN = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, xTestImages.shape[1], xTestImages.shape[2], xTestImages.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 18, 'nonlinearity': softmax}),
]

convolutionalNeuralNet = NeuralNet(
    layers=layersOfCNN,
    max_epochs=10,

    update=adam,
    update_learning_rate=0.0002,

    objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)
convolutionalNeuralNet.fit(xTestImages, y)
