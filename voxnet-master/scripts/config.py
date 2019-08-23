import lasagne
import voxnet
from voxnet import activations
import theano.tensor as T

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
cfg = {
    'dims': (32, 32, 32),
    'batch_size': 32,
    'learning_rate': lr_schedule,
    'reg': 0.001,
    'momentum': 0.9,
    'batches_per_chunk': 64,
    'n_channels': 1,
    'n_classes': 10,
    'n_levels': 32,
    'n_rings': 90,
    'dim_value': 3,
    'stride_z': 1,
    'max_epochs': 80,
    'max_jitter_ij': 2,
    'max_jitter_k': 2,
    'n_rotations': 12,
    'checkpoint_every_nth': 4000,

}


def get_model():
    dims, n_channels, n_classes, n_levels, n_rings, dim_value = \
        tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes'], cfg['n_levels'], cfg['n_rings'], cfg['dim_value']
    batch_size = cfg['batch_size']
    shape = (None, n_channels, n_rings, 3)
    l_in = lasagne.layers.InputLayer(shape=shape)

    '''    
    l_conv1=lasagne.layers.Conv1DLayer(
            incoming=l_in,
            num_filters=1,
            filter_size=3,
            stride=3,
            pad='valid'
        )
    '''

    '''
    Ring level feature abstracted
    '''
    l_conv1 = lasagne.layers.Conv2DLayer(
        incoming=l_in,
        num_filters=32,
        filter_size=(5, 3),
        stride=(5, 1),
        pad='valid',
        name='conv1',
        W=voxnet.init.Prelu(),
        # nonlinearity=activations.leaky_relu_001

    )
    l_conv2 = lasagne.layers.Conv2DLayer(
        incoming=l_conv1,
        num_filters=32,
        filter_size=(3, 1),
        stride=(3, 1),
        pad='valid',
        name='conv2',
        W=voxnet.init.Prelu(),
        # nonlinearity = activations.leaky_relu_001
    )
    l_conv3 = lasagne.layers.Conv2DLayer(
        incoming=l_conv2,
        num_filters=32,
        filter_size=(3, 1),
        stride=(3, 1),
        pad='valid',
        name='conv3',
        W=voxnet.init.Prelu(),
        # nonlinearity = activations.leaky_relu_001
    )
    l_conv4 = lasagne.layers.Conv2DLayer(
        incoming=l_conv3,
        num_filters=32,
        filter_size=(2, 1),
        stride=(2, 1),
        pad='valid',
        name='conv4',
        W=voxnet.init.Prelu(),
        # nonlinearity=activations.leaky_relu_001
    )
    l_shape1 = lasagne.layers.reshape(
        incoming=l_conv4,
        shape=((-1, 1, 32)),
        name='shape1'
    )
    l_pool1 = lasagne.layers.MaxPool1DLayer(
        incoming=l_shape1,
        pool_size=2,
        name='pool1'
    )
    l_fc1 = lasagne.layers.DenseLayer(
        incoming=l_pool1,
        num_units=128,
        W=lasagne.init.Normal(std=0.01),
        name='fc1'

    )
    l_fc2 = lasagne.layers.DenseLayer(
        incoming=l_fc1,
        num_units=n_classes,
        W=lasagne.init.Normal(std=0.01),
        # nonlinearity=T.nnet.sigmoid,
        name='fc2'

    )
    # l_bn1 = lasagne.layers.BatchNormLayer(
    #     incoming= l_fc2,
    #     name= 'bn1'
    # )

    '''
    Model level feature abstracted
    '''
    l_shape2 = lasagne.layers.reshape(
        incoming=l_fc2,
        shape=((batch_size, -1)),
        name='shape2'
    )
    l_fc3 = lasagne.layers.DenseLayer(
        incoming=l_shape2,
        num_units=128,
        W=lasagne.init.Normal(std=0.01),
        name='fc3'

    )
    l_fc4 = lasagne.layers.DenseLayer(
        incoming=l_fc3,
        num_units=32,
        W=lasagne.init.Normal(std=0.01),
        name='fc4'

    )
    l_fc5 = lasagne.layers.DenseLayer(
        incoming=l_fc4,
        num_units=n_classes,
        W=lasagne.init.Normal(std=0.01),
        name='fc5'

    )
    l_shape3 =lasagne.layers.reshape(
        incoming=l_fc5,
        shape=((batch_size, -1)),
        name='shape2'
    )
    return {'l_in': l_in, 'l_out': l_fc5}
