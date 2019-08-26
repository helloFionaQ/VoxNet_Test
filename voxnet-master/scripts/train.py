# -*- coding: utf-8 -*-



import argparse
import imp
import time
import logging

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

import voxnet
from voxnet import npytar

#import pyvox

def make_training_functions(cfg, model):

    # 这里l_out是model的输出层
    l_out = model['l_out']
    #声明一个Batch——index的变量
    batch_index = T.iscalar('batch_index')
    # bct01
    #x是五维向量 y是一维
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
    #log.info('output_shape = {}'.format(out_shape))

    # 切片函数 参数：start ，stop[step] 用法arr[batch_slice]
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])

    # 用给定的输入和网络模型 做输出
    out = lasagne.layers.get_output(l_out, X)
    # 也用来做输出，但会屏蔽掉所有的drop-out层
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)
    # 获取训练网络的所有的参数 一般用于更新网络表达式
    params = lasagne.layers.get_all_params(l_out)
    l2_norm = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)
    # 判断 x是不是某类型 （实例，类型名） dict 字典
    if isinstance(cfg['learning_rate'], dict):
        # share将变量共享为全局变量 ，在多个函数中公用
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))

    softmax_out = T.nnet.softmax( out )
    loss = T.cast(T.mean(T.nnet.categorical_crossentropy(softmax_out, y)), 'float32')
    pred = T.argmax( dout, axis=1 )
    error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )
    # 正则化损失函数 l2使权值足够小
    reg_loss = loss + cfg['reg']*l2_norm
    # 动量梯度下降 更新params
    updates = lasagne.updates.momentum(reg_loss, params, learning_rate, cfg['momentum'])


    #shared相当于一个全局变量
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)

    update_iter = theano.function([batch_index], reg_loss,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })

    error_rate_fn = theano.function([batch_index], error_rate, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
    tfuncs = {'update_iter':update_iter,
             'error_rate':error_rate_fn,
             'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars


#抖动加强
def jitter_chunk(src, cfg):
    dst = src.copy()
    # 二项分布
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst

# 读取数据
def data_loader(cfg, fname):

    dims = cfg['dims']
    # the number for reading each time
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            xc = jitter_chunk(xc, cfg)
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    if len(yc) > 0:
        # pad to nearest multiple of batch_size
        if len(yc)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(yc):] = xc[:(new_size-len(yc))]
            yc = yc + yc[:(new_size-len(yc))]

        xc = jitter_chunk(xc, cfg)
        yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))

def main(args):
    # 加载配置文件
    config_module = imp.load_source('config', args.config_path)
    # cfg文件记录属性
    cfg = config_module.cfg

    # model里面是卷积层等网络模型
    model = config_module.get_model()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(args.metrics_fname))

    mlog = voxnet.metrics_logging.MetricsLogger(args.metrics_fname, reinitialize=True)

    logging.info('Compiling theano functions...')
    tfuncs, tvars = make_training_functions(cfg, model)

    logging.info('Training...')
    itr = 0
    last_checkpoint_itr = 0
    loader = (data_loader(cfg, args.training_fname))

    # 设置迭代
    # xrange的用法与range相同，即xrange([start,] stop[, step])根据start与stop指定的范围以及step设定的步长,
    # 他所不同的是xrange并不是生成序列，而是作为一个生成器。即他的数据生成一个取出一个。
    for epoch in xrange(cfg['max_epochs']):
        loader = (data_loader(cfg, args.training_fname))

        for x_shared, y_shared in loader:
            num_batches = len(x_shared)//cfg['batch_size']
            tvars['X_shared'].set_value(x_shared, borrow=True)
            tvars['y_shared'].set_value(y_shared, borrow=True)
            lvs,accs = [],[]
            for bi in xrange(num_batches):
                lv = tfuncs['update_iter'](bi)
                lvs.append(lv)
                acc = 1.0-tfuncs['error_rate'](bi)
                accs.append(acc)
                itr += 1
            loss, acc = float(np.mean(lvs)), float(np.mean(acc))
            logging.info('epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc))
            mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)

            if isinstance(cfg['learning_rate'], dict) and itr > 0:
                keys = sorted(cfg['learning_rate'].keys())
                new_lr = cfg['learning_rate'][keys[np.searchsorted(keys, itr)-1]]
                lr = np.float32(tvars['learning_rate'].get_value())
                if not np.allclose(lr, new_lr):
                    logging.info('decreasing learning rate from {} to {}'.format(lr, new_lr))
                    tvars['learning_rate'].set_value(np.float32(new_lr))
            if itr-last_checkpoint_itr > cfg['checkpoint_every_nth']:
                voxnet.checkpoints.save_weights('weights.npz', model['l_out'],
                                                {'itr': itr, 'ts': time.time()})
                last_checkpoint_itr = itr


    logging.info('training done')
    voxnet.checkpoints.save_weights('weights1.npz', model['l_out'],
                                    {'itr': itr, 'ts': time.time()})



# 尝试取消控制台输入
class noneParser():
    config_path ='config/shapenet10.py '
    training_fname ='shapenet10_train.tar'




if __name__=='__main__':

    parser = argparse.ArgumentParser()

    #  python train.py config/shapenet10.py shapenet10_train.tar
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('training_fname', type=Path, help='training .tar file')
    parser.add_argument('--metrics-fname', type=Path, default='metrics.jsonl', help='name of metrics file')

    args = parser.parse_args()

    # args=noneParser()
    main(args)

