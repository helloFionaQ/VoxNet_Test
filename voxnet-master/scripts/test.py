# -*- coding: utf-8 -*-

import imp
import argparse
import logging

from path import Path
import numpy as np
import theano
import theano.tensor as T
import lasagne

from sklearn import metrics as skm

import voxnet

# npy tar的类型转换
from statsmodels.sandbox.distributions.examples.matchdist import high
from voxnet import npytar

exception_verbosity=high

def make_test_functions(cfg, model):
    l_out = model['l_out']
    batch_index = T.iscalar('batch_index')
    # bct01
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
    #log.info('output_shape = {}'.format(out_shape))

    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    out = lasagne.layers.get_output(l_out, X)
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)

    params = lasagne.layers.get_all_params(l_out)

    softmax_out = T.nnet.softmax( out )
    pred = T.argmax( dout, axis=1 )

    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)

    tfuncs = {'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
            }
    return tfuncs, tvars

#感觉像是把体素数据转换成卷积操作数据 转化成对应数据-标签的形式
def data_loader(cfg, fname):
    dims = cfg['dims']
    chunk_size = cfg['n_rotations']
    # 第一个参数 shape 是一个元组 +代表元组的拼接
    # 如果元组的元素只有一个值的时候需要在值的后面加上逗号，否则括号会被误识别
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    #ix 是index （x,name)是读取的元组 x是模型数据 name是文件名
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        # yc是类别编号
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            # yield相当于一个高级return，与next，send结合
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    # assert 用于抛异常
    assert(len(yc)==0)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
    model = config_module.get_model()

    logging.info('Loading weights from {}'.format(args.weights_fname))
    # 加载权重
    voxnet.checkpoints.load_weights(args.weights_fname, model['l_out'])

    loader = (data_loader(cfg, args.testing_fname))

    tfuncs, tvars = make_test_functions(cfg, model)

    yhat, ygnd = [], []
    for x_shared, y_shared in loader:
        pred = np.argmax(np.sum(tfuncs['dout'](x_shared), 0))
        yhat.append(pred)
        ygnd.append(y_shared[0])
        #assert( np.max(y_shared)==np.min(y_shared)==y_shared[0] )

    yhat = np.asarray(yhat, dtype=np.int)
    ygnd = np.asarray(ygnd, dtype=np.int)

    acc = np.mean(yhat==ygnd).mean()
    macro_recall = skm.recall_score(yhat, ygnd, average='macro')
    print('normal acc = {}, macro_recall = {}'.format(acc, macro_recall))

    if args.out_fname is not None:
        logging.info('saving predictions to {}'.format(args.out_fname))
        np.savez_compressed(args.out_fname, yhat=yhat, ygnd=ygnd)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('testing_fname', type=Path, nargs='?', default='shapenet10_test.tar')
    parser.add_argument('--weights-fname', type=Path, default='weights.npz')
    parser.add_argument('--out-fname', type=Path, default=None, help='Save output to this file. Format is .npz')
    args = parser.parse_args()
    main(args)
