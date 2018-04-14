# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-28 上午10:04
# @Author  : wang shen
# @web    : 
# @File    : crf.py

from torch_file.main import loadDataset
from torch.autograd import Variable
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm
import pycrfsuite
import config
import torch
import h5py


def save_inputs(model, loader, size, dir, config):
    print('Saving inputs')
    f = h5py.File(dir, 'w')

    f.create_dataset('x', (size, config.max_length, config.q_o_state_size))
    f.create_dataset('y', (size, config.max_length), 'i')
    f.create_dataset('z', (size, 1), 'i')

    n = 0
    for batch_id, (q, q_id, q_mask, q_o, q_o_id, q_o_mask, c, c_mask, label) in enumerate(loader):

        q = Variable(q.long())
        q_mask = Variable(q_mask.byte())
        q_o = Variable(q_o.long())
        q_o_mask = Variable(q_o_mask.byte())
        c = Variable(c.long())
        c_mask = Variable(c_mask.byte())
        label = Variable(label.long(), requires_grad=False)

        q_o_out = model.get_lstm(q, q_o, c, q_mask, q_o_mask, c_mask)
        q_o_out = q_o_out.data.cpu().tolist()
        q_o_l = q_o_mask.data.eq(0).long().sum(1).squeeze()

        label = label.data.cpu().tolist()
        label = [e * config.max_length for e in label]

        for o, l, e in zip(q_o_out, label, q_o_l):
            f['x'][n] = o
            f['y'][n] = l
            f['z'][n] = e
            n += 1

    print('num :', n)

    f.close()

    return n


def load_inputs(n, dir, config):
    print('Load inputs')
    f = h5py.File(dir)
    x = f['x'][:n]
    y = f['y'][:n]
    z = f['z'][:n]
    f.close()

    x_s = []
    y_s = []
    z_s = []
    fs = [str(i) for i in range(config.q_o_state_size)]

    for o, l, e in tqdm(zip(x, y, z)):
        s = []
        for v in o:
            s.append(dict(zip(fs, v)))
        x_s.append(s)

        l = [str(t) for t in l]
        y_s.append(l)

        z_s.append(e)

    print(' y  : ( ', len(y_s), ', ', len(y_s[0]), ')')

    return x_s, y_s, z_s


def train_crf(x_s, y_s, d):
    print('train crf')
    t = pycrfsuite.Trainer(verbose=False)

    for x, y in zip(x_s, y_s):
        t.append(x, y)

    t.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 500,
        'feature.possible_transitions': True
    })

    print('start train crf')
    t.train(d)
    print('end train crf')


def get_tags(x_s, y_s, z_s, d, d_s, d_t):
    print('test crf')
    t = pycrfsuite.Tagger()
    t.open(d)

    f_s = open(d_s, 'rb')
    f_t = open(d_t, 'wb')

    for x, y, z in zip(x_s, y_s, z_s):
        p_t = t.tag(x)
        p_t = [int(s) for k, s in enumerate(p_t) if k < z[0]]
        # print('Predicted:', Counter(p_t).most_common(1)[0][0], 'Correct: ', y[0])

        l = f_s.readline().decode('utf8').split()
        l[4] = 'true' if Counter(p_t).most_common(1)[0][0] else 'false'
        s = l[0] + '	' + l[1] + '	' + l[2] + '	' + l[3] + '	' + l[4] + '\n'
        f_t.write(s.encode('utf8'))

    f_s.close()
    f_t.close()


if __name__ == '__main__':
    config = config.get_args()

    train_dataset = loadDataset(config.train_data_h5)
    val_dataset = loadDataset(config.valid_data_h5)
    test_dataset_2016 = loadDataset(config.test_data_h5_2016)
    test_dataset_2017 = loadDataset(config.test_data_h5_2017)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    test_loader_2016 = torch.utils.data.DataLoader(test_dataset_2016, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader_2017 = torch.utils.data.DataLoader(test_dataset_2017, batch_size=config.batch_size, num_workers=config.num_workers)

    model = torch.load(config.load_model_dir)

    train_set_s = train_dataset.__len__()
    d = config.crf_train_dir
    n = save_inputs(model, train_loader, train_set_s, d, config)
    # n = 1999
    x, y, z = load_inputs(n, d, config)
    train_crf(x, y, config.crf_dir)

    test_set_s = test_dataset_2016.__len__()
    d = config.crf_test_2016_dir
    n = save_inputs(model, test_loader_2016, test_set_s, d, config)
    x, y, z = load_inputs(n, d, config)
    get_tags(x, y, z, config.crf_dir, config.result_data_file_2016, config.result_data_crf_2016)

    test_set_s = test_dataset_2017.__len__()
    d = config.crf_test_2017_dir
    n = save_inputs(model, test_loader_2017, test_set_s, d, config)
    x, y, z = load_inputs(n, d, config)
    get_tags(x, y, z, config.crf_dir, config.result_data_file_2017, config.result_data_crf_2017)

