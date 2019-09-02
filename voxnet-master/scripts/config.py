import lasagne
import theano.tensor as T
lr_schedule3D = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
lr_schedule2D = { 0: 0.002,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
lr_schedule1D = { 0: 0.002,
                60000: 0.001,
                400000: 0.0005,
                600000: 0.0001,
                }
lr_schedule=lr_schedule2D
metric =[
    {'dims' : 1,
    'test_fname' : r'/home/haha/Documents/PythonPrograms/VoxNet_Test/voxnet-master/scripts/ring_datas/ringdata_1D_test.tar',
    'train_fname': r'/home/haha/Documents/PythonPrograms/VoxNet_Test/voxnet-master/scripts/ring_datas/ringdata_1D_train.tar',
    'weight_fname' : 'weights_1D.npz',
    'metric_fname' :'metrics_1D.jsonl'},

    {'dims' : 2,
    'test_fname' : r'/home/haha/Documents/PythonPrograms/VoxNet_Test/voxnet-master/scripts/ring_datas/ringdata_2D_test.tar',
    'train_fname': r'/home/haha/Documents/PythonPrograms/VoxNet_Test/voxnet-master/scripts/ring_datas/ringdata_2D_train.tar',
    'weight_fname' : 'weights_2D.npz',
    'metric_fname' :'metrics_2D.jsonl'},

    {'dims' : 3,
    'test_fname' : r'/home/haha/Documents/PythonPrograms/Ring_Conv/ring_data_1D_test.tar',
    'train_fname': r'/home/haha/Documents/PythonPrograms/Ring_Conv/ring_data_level15_nj_dis.tar',
    'weight_fname' : 'weights_nojitter_1D.npz',
    'metric_fname' :'metrics_nojitter_1D.jsonl'},
]
cfg={
    'batch_size' : 32,
    'batches_per_chunk': 64,
    'n_channels' : 1,
    'n_classes' : 10,
    'n_levels': 15,
    'n_rings' : 99,
    'dim_value' : 3,
    'stride_z' : 1,
    'learning_rate' : lr_schedule,
    'reg' : 0.001,
    'momentum' : 0.9,
    'max_epochs' : 80,
    'max_jitter_ij' : 2,
    'max_jitter_k' : 2,
    'n_rotations' : 12,
    'checkpoint_every_nth' : 4000,

   ' n_filters':64,

}

conv_size={
    'size90':[1,5,3,3,2],
    'size99':[1,3,3,3,3]

}

def get_model():
    n_channels, n_classes, n_levels, n_rings , dim_value = \
         cfg['n_channels'], cfg['n_classes'],cfg['n_levels'], cfg['n_rings'], cfg['dim_value']
    batch_size = cfg['batch_size']

    dims=metric[1]['dims'] # change the index when changing input dims
    n_filters=cfg[' n_filters']
    shape = (None, 1, n_levels,n_rings, dims)
    fw=conv_size['size99']

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
    l_shape =lasagne.layers.reshape(
        incoming=l_in,
        shape= ((-1, 1,n_rings,dims)),
        name= 'shape'
    )

    l_conv=lasagne.layers.Conv2DLayer(
        incoming=l_shape,
        num_filters=n_filters,
        filter_size=(1, dims),
        stride=(1,1),
        pad='valid',
        name= 'conv0',
        # nonlinearity=activations.leaky_relu_001

    )

    l_conv1=lasagne.layers.Conv2DLayer(
        incoming=l_conv,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name= 'conv1',
        # nonlinearity=activations.leaky_relu_001

    )

    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_conv1,
        p = 0.2,
        name = 'drop1'
        )
    l_conv2=lasagne.layers.Conv2DLayer(
        incoming=l_drop1,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv2',
        # nonlinearity = activations.leaky_relu_001
    )

    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_conv2,
        p = 0.3,
        name = 'drop2',
        )
    l_conv3=lasagne.layers.Conv2DLayer(
        incoming=l_drop2,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv3',
        # nonlinearity = activations.leaky_relu_001
    )

    l_conv4=lasagne.layers.Conv2DLayer(
        incoming=l_conv3,
        num_filters=2*n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv4',
        # nonlinearity=activations.leaky_relu_001
    )
    l_conv5=lasagne.layers.Conv2DLayer(
        incoming=l_conv4,
        num_filters=512,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv5',
        # nonlinearity=activations.leaky_relu_001
    )
    l_ds1=lasagne.layers.DimshuffleLayer(
        incoming=l_conv5,
        pattern=(0,3,1,2),
        name='ds1'
    )
    l_pool1=lasagne.layers.MaxPool2DLayer(
        incoming=l_ds1,
        pool_size=(512,1),
        name='pool1'
    )
    # -------------shape n*1*nrings------------------------

    '''
    Model level feature abstracted
    '''
    l_shape2 = lasagne.layers.reshape(
        incoming=l_pool1,
        shape=((-1,n_levels,n_rings)),
        name='shape2'
    )
    # l_conv11=lasagne.layers.Conv2DLayer(
    #     incoming=l_shape2,
    #     num_filters=n_filters,
    #     filter_size=(3, 1),
    #     stride=(1,1),
    #     pad='valid',
    #     name='conv11',
    #     # nonlinearity=activations.leaky_relu_001
    # )
    l_fc10 = lasagne.layers.DenseLayer(
        incoming=l_shape2,
        num_units=1024,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc10',

    )
    l_fc11 = lasagne.layers.DenseLayer(
        incoming=l_fc10,
        num_units=512,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc11',

    )
    l_fc12 = lasagne.layers.DenseLayer(
        incoming=l_fc11,
        num_units=128,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc12'

    )
    l_fc13 = lasagne.layers.DenseLayer(
        incoming=l_fc12,
        num_units=n_classes,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc13'

    )

    return {'l_in':l_in,'l_out':l_fc13}