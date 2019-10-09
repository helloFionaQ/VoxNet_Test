# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
from sklearn import metrics as skm

import voxnet
from scripts.model import config

from voxnet import npytar

dims= config.cfg['value_dim']
train_fname = config.cfg['train_fname']
test_fname = config.cfg['test_fname']
weight_name= config.cfg['weight_fname']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = config.cfg['log_dir']
WEIGHT_FILE=os.path.join(LOG_DIR,weight_name)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
NOTE=config.NOTE
# 加载配置文件
cfg = config.cfg
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
out_fname=None
def make_test_functions(cfg, model):
    l_out = model['l_out']
    l_test = model['l_test']
    batch_index = T.iscalar('batch_index')
    # bct01
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)

    #log.info('output_shape = {}'.format(out_shape))

    test=lasagne.layers.get_output_shape(l_test)

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
              'test' : test,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
            }
    return tfuncs, tvars

#感觉像是把体素数据转换成卷积操作数据 转化成对应数据-标签的形式
def data_loader(cfg):
    chunk_size = cfg['n_rotations']

    # 第一个参数 shape 是一个元组 +代表元组的拼接
    # 如果元组的元素只有一个值的时候需要在值的后面加上逗号，否则括号会被误识别
    xc = np.zeros((chunk_size, cfg['n_channels'],cfg['n_levels'],cfg['n_rings'],dims), dtype=np.float32)
    reader = npytar.NpyTarReader(test_fname)
    yc = []
    #ix 是index （x,name)是读取的元组 x是模型数据 name是文件名
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        # yc是类别编号
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            # yield相当于一个高级return，与next，send结合
            yield (xc, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    # assert 用于抛异常
    assert(len(yc)==0)


def main():
    cfg = config.cfg
    model = config.get_model()
    voxnet.checkpoints.load_weights(WEIGHT_FILE, model['l_out'])
    ygs=np.zeros((10,))
    yhs=np.zeros((10,))
    loader = (data_loader(cfg))
    log_string(NOTE)
    log_string ('Testing file : %s'%os.path.split(test_fname)[-1])
    print 'Compiling theano functions...'
    tfuncs, tvars = make_test_functions(cfg, model)
    print 'Testing...'
    start_time = time.time()
    yhat, ygnd = [], []

    for x_shared, y_shared in loader:
        # pred = np.argmax(np.sum(tfuncs['dout'](x_shared), 0))
        # yhat.append(pred)
        # ygnd.append(y_shared[0])

        pred=tfuncs['pred'](x_shared)
        for i,yg in enumerate(y_shared):
            yg=int(yg)
            ygs[yg]+=1
            if yg==pred[i]:
                yhs[yg]+=1

        ygnd.extend(y_shared)
        yhat.extend(pred)


    yhat = np.asarray(yhat, dtype=np.int)
    ygnd = np.asarray(ygnd, dtype=np.int)
    yhat.reshape((-1))
    ygnd.reshape((-1))

    accs_cls =[]
    for i in range(10):
        accs_cls.append((float(yhs[i])/float(ygs[i])))

    acc = np.mean(yhat==ygnd).mean()
    acc_cls = float(np.mean(accs_cls))
    macro_recall = skm.recall_score(yhat, ygnd, average='macro')
    log_string('normal acc = {}, macro_recall = {}'.format(acc, macro_recall))
    log_string('class acc = {}'.format(acc_cls))
    for i in range(10):
        log_string ('   %s : %f' % (config.name_dic[i + 1], (float(yhs[i]) / float(ygs[i]))))
    end_time = time.time()
    log_string('running time : %.3f secs'%(end_time - start_time))

    if out_fname is not None:
        logging.info('saving predictions to {}'.format(out_fname))
        np.savez_compressed(out_fname, yhat=yhat, ygnd=ygnd)


if __name__=='__main__':
    main()
    LOG_FOUT.close()
