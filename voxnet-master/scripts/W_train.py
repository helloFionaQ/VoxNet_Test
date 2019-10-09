# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import voxnet
# from scripts.model import config
from scripts.model import config_trail as config
from voxnet import npytar
from sklearn import metrics as skm

dims= config.cfg['value_dim']
train_fname = config.cfg['train_fname']
test_fname = config.cfg['test_fname']
weight_name= config.cfg['weight_fname']
metric_fname= config.cfg['metric_fname']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = config.cfg['log_dir']
MODEL_FILE = os.path.join(BASE_DIR,'model', 'config_trail.py')
WEIGHT_FILE=os.path.join(LOG_DIR,weight_name)
METRIC_FILE=os.path.join(LOG_DIR,metric_fname)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp W_train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
NOTE=config.NOTE
# 加载配置文件
cfg = config.cfg
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def make_training_functions(model):

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
    # 判断 x是不是某类型 （实例，类型名） dict 字典tensorboard
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
    pred_fn=theano.function([batch_index], pred, givens={
            X: X_shared[batch_slice]
            ,
        })

    update_iter = theano.function([batch_index], reg_loss,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })

    error_rate_fn = theano.function([batch_index], error_rate, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })

    loss_fn = theano.function([batch_index], reg_loss, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
    tfuncs = {'update_iter':update_iter,
             'error_rate':error_rate_fn,
             'loss':loss_fn,
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

# 读取数据
def data_loader(fname):
    # the number for reading each time
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],cfg['n_levels'],cfg['n_rings'],dims), dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            yield (xc, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    if len(yc) > 0:
        # pad to nearest multiple of batch_size

        new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
        xc = xc[:new_size]
        xc[len(yc):] = xc[:(new_size-len(yc))]
        yc = yc + yc[:(new_size - len(yc))]
        yield (xc, np.asarray(yc, dtype=np.float32))


def main():
    x_i = []  # 定义一个 x 轴的空列表用来接收动态的数据
    y_l = []  # 定义一个 y 轴的空列表用来接收动态的数据
    y_a = []  # 定义一个 y 轴的空列表用来接收动态的数据

    plt.ion()

    # model里面是卷积层等网络模型
    model = config.get_model()
    mlog = voxnet.metrics_logging.MetricsLogger(metric_fname, reinitialize=True)
    log_string (NOTE)
    log_string ('Training file : %s'%os.path.split(train_fname)[-1])
    log_string ('Compiling theano functions...')
    tfuncs, tvars = make_training_functions(model)
    log_string( 'Training...')
    start_time=time.time()
    itr = 0
    last_checkpoint_itr = 0

    # 设置迭代
    # xrange的用法与range相同，即xrange([start,] stop[, step])根据start与stop指定的范围以及step设定的步长,
    # 他所不同的是xrange并不是生成序列，而是作为一个生成器。即他的数据生成一个取出一个。
    for epoch in xrange(cfg['max_epochs']):
        loader = (data_loader(train_fname))

        for x_shared, y_shared in loader:
            num_batches = len(x_shared)//cfg['batch_size']
            tvars['X_shared'].set_value(x_shared, borrow=True)
            tvars['y_shared'].set_value(y_shared, borrow=True)
            lvs, accs = [],[]
            for bi in xrange(num_batches):
                lv = tfuncs['update_iter'](bi)
                lvs.append(lv)
                acc = 1.0-tfuncs['error_rate'](bi)
                accs.append(acc)
                itr += 1
            loss, acc = float(np.mean(lvs)), float(np.mean(accs))
            log_string ('epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc))
            mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)
            y_l.append(loss)
            y_a.append(acc)
            x_i.append(itr)


            if isinstance(cfg['learning_rate'], dict) and itr > 0:
                keys = sorted(cfg['learning_rate'].keys())
                new_lr = cfg['learning_rate'][keys[np.searchsorted(keys, itr)-1]]
                lr = np.float32(tvars['learning_rate'].get_value())
                if not np.allclose(lr, new_lr):
                    log_string( 'decreasing learning rate from {} to {}'.format(lr, new_lr))
                    tvars['learning_rate'].set_value(np.float32(new_lr))
            if itr-last_checkpoint_itr > cfg['checkpoint_every_nth']:
                voxnet.checkpoints.save_weights(WEIGHT_FILE, model['l_out'],
                                                {'itr': itr, 'ts': time.time()})
                log_string( 'Model saved')
                test(tvars,tfuncs)
                # drawer(plt, y_l, y_a, x_i)
                last_checkpoint_itr = itr

    log_string( 'training done')
    end_time=time.time()
    log_string( 'running time : %.3f hours '%((end_time-start_time)/60.0/60.0))
    voxnet.checkpoints.save_weights(weight_name, model['l_out'],
                                    {'itr': itr, 'ts': time.time()})


def test(tvars,tfuncs):
    log_string('Testing on checkpoint')
    start_time = time.time()
    loader = (data_loader(test_fname))
    yhat, ygnd = [], []
    ygs = np.zeros((10,))
    yhs = np.zeros((10,))

    for x_shared, y_shared in loader:
        num_batches = len(x_shared) // cfg['batch_size']
        lvs, accs = [], []
        tvars['X_shared'].set_value(x_shared, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)
        for bi in xrange(num_batches):
            lv = tfuncs['loss'](bi)
            lvs.append(lv)
            acc = 1.0 - tfuncs['error_rate'](bi)
            accs.append(acc)
            pred = tfuncs['pred'](bi)
            yhat.extend(pred)
        ygnd.extend(y_shared)

        for i,yg in enumerate(y_shared):
            yg=int(yg)
            ygs[yg]+=1
            if yg==pred[i]:
                yhs[yg]+=1
        for bi in xrange(num_batches):
            lv = tfuncs['loss'](bi)
            lvs.append(lv)
            acc = 1.0 - tfuncs['error_rate'](bi)
            accs.append(acc)
    loss, acc = float(np.mean(lvs)), float(np.mean(accs))
    log_string('New result:')
    log_string('loss: {}, acc: {}'.format(loss, acc))
    accs_cls =[]
    for i in range(10):
        accs_cls.append((float(yhs[i])/float(ygs[i])))

    acc = np.mean(yhat==ygnd).mean()
    acc_cls = float(np.mean(accs_cls))
    log_string('Traditional result:')
    macro_recall = skm.recall_score(yhat, ygnd, average='macro')
    log_string('normal acc = {}, macro_recall = {}'.format(acc, macro_recall))
    log_string('class acc = {}'.format(acc_cls))
    for i in range(10):
        log_string ('   %s : %f' % (config.name_dic[i + 1], (float(yhs[i]) / float(ygs[i]))))
    end_time = time.time()
    log_string('running time : %.3f secs'%(end_time - start_time))



if __name__=='__main__':

    main()
    LOG_FOUT.close()

