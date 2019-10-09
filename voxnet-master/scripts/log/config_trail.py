import lasagne
import scripts.utils.activations
lr_schedule3D = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
lr_schedule = { 0: 0.002,
                60000: 0.0001,
                400000: 0.0001,
                600000: 0.00001,
                }


cfg={
    'test_fname': r'/home/haha/Documents/PythonPrograms/datasets/ringdata_02_test.tar',
    # 'train_fname': r'/home/haha/Documents/PythonPrograms/datasets/ringdata_05_train_15993j.tar',
    'train_fname': r'/home/haha/Documents/PythonPrograms/datasets/ringdata_02_train.tar',
    'weight_fname': 'weights_2D.npz',
    'metric_fname': 'metrics_2D.jsonl',
    'log_dir':'log',
    'value_dim':3,
    'batch_size' : 32,
    'batches_per_chunk': 64,
    'n_channels' : 1,
    'n_classes' : 10,
    'n_levels': 15,
    'n_rings' : 99,
    'stride_z' : 1,
    'learning_rate' : lr_schedule,
    'reg' : 0.001,
    'momentum' : 0.9,
    # 'max_epochs' : 40,
    'max_epochs' : 80,
    'max_jitter_ij' : 2,
    'max_jitter_k' : 2,
    'n_rotations' : 12,
    'checkpoint_every_nth' : 4000,

   ' n_filters':64,
}

name_dic = {
    1: "bathtub",
    2: "bed",
    3: "chair",
    4: "desk",
    5: "dresser",
    6: "monitor",
    7: "night_stand",
    8: "sofa",
    9: "table",
    10: "toilet"
}

NOTE='Model shape: %s * %s * %s, Network : 6conv(512) 1fc(512) 4fc' % (cfg['n_levels'],cfg['n_rings'],cfg['value_dim'])

def get_model():
    n_channels, n_classes, n_levels, n_rings , dim_value = \
         cfg['n_channels'], cfg['n_classes'],cfg['n_levels'], cfg['n_rings'], cfg['value_dim']
    batch_size = cfg['batch_size']

    n_filters=cfg[' n_filters']
    shape = (None, 1, n_levels,n_rings, dim_value)

    l_in = lasagne.layers.InputLayer(shape=shape)
    '''
    Ring level feature abstracted
    '''
    l_shape =lasagne.layers.reshape(
        incoming=l_in,
        shape= ((-1, 1,n_rings,dim_value)),
        name= 'shape'
    )

    l_conv1=lasagne.layers.Conv2DLayer(
        incoming=l_shape,
        num_filters=n_filters,
        filter_size=(1, dim_value),
        stride=(1,1),
        pad='valid',
        name= 'conv1',
        nonlinearity=scripts.utils.activations.leaky_relu_001
    )

    #
    l_conv2=lasagne.layers.Conv2DLayer(
        incoming=l_conv1,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name= 'conv2',
        nonlinearity=scripts.utils.activations.leaky_relu_001

    )

    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_conv2,
        p = 0.2,
        name = 'drop1'
        )
    l_conv3=lasagne.layers.Conv2DLayer(
        incoming=l_drop1,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv3',
        nonlinearity = scripts.utils.activations.leaky_relu_001
    )

    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_conv3,
        p = 0.3,
        name = 'drop2',
        )
    l_conv4=lasagne.layers.Conv2DLayer(
        incoming=l_drop2,
        num_filters=n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv4',
        nonlinearity = scripts.utils.activations.leaky_relu_001
    )

    l_conv5=lasagne.layers.Conv2DLayer(
        incoming=l_conv4,
        num_filters=2*n_filters,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv5',
        nonlinearity=scripts.utils.activations.leaky_relu_001
    )
    l_conv6=lasagne.layers.Conv2DLayer(
        incoming=l_conv5,
        num_filters=512,
        filter_size=(1, 1),
        stride=(1,1),
        pad='valid',
        name='conv6',
        nonlinearity=scripts.utils.activations.leaky_relu_001
    )
    # (None, 512, 99, 1)
    l_shape3 = lasagne.layers.reshape(
        incoming=l_conv6,
        shape=((-1,1,512,n_rings)),
        name='shape3'
    )
    #
    # l_ds1=lasagne.layers.DimshuffleLayer(
    #     incoming=l_conv6,
    #     pattern=(0,3,1,2),
    #     name='ds1'
    # )  # shape: (None, 1, 512,99 )
    l_pool1=lasagne.layers.MaxPool2DLayer(
        incoming=l_shape3,
        pool_size=(512,1),
        name='pool1'
    ) # shape : (None, 1,  1,99)

    # -------------shape n*1*nrings------------------------

    '''
    Model level feature abstracted
    '''
    l_shape2 = lasagne.layers.reshape(
        incoming=l_pool1,
        shape=((-1,n_levels,n_rings,1)),
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
    l_fc1 = lasagne.layers.DenseLayer(
        incoming=l_shape2,
        num_units=1024,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc1',

    )
    l_fc2 = lasagne.layers.DenseLayer(
        incoming=l_fc1,
        num_units=512,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc2',

    )
    l_fc3 = lasagne.layers.DenseLayer(
        incoming=l_fc2,
        num_units=128,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc3'

    )
    l_fc4 = lasagne.layers.DenseLayer(
        incoming=l_fc3,
        num_units=n_classes,
        W=lasagne.init.Normal(std=0.01),
        name= 'fc4'

    )


    return {'l_in':l_in,'l_out':l_fc4, 'l_test':l_conv6}