import train
import numpy as np
from voxnet import npytar



cfg = {'batch_size' : 32,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' : 80,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 4000
       }
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

        yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))

for epoch in xrange(cfg['max_epochs']):
    loader = (data_loader(cfg, 'shapenet10_train.tar'))
    for x,y in loader :
        print 'hellp'