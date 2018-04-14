# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-24 下午4:39
# @Author  : wang shen
# @web    : 
# @File    : train.py

import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def save_model(model, epoch, loss, acc, dir):
    model_path = dir + '/' + 'loss' + str(round(loss, 4)) + '_acc' + str(round(acc, 4)) + '_' + str(epoch)
    with open(model_path, 'wb') as f:
        torch.save(model, f)


def train(model, train_loader, config):
    print('Train model')
    para = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(para, lr=config.learning_rate)

    acc = 0.75
    for e in range(config.num_epochs):
        l, a = train_epoch(model, e, train_loader, opt)

        if a >= acc:
            acc = a
            save_model(model, e, l, a, config.model_dir)

    print('End model')


def train_epoch(model, epoch, loader, opt):
    print('Train epoch :', epoch)
    model.train()

    e_loss = 0.0
    n_b = 0
    l_l = []
    p_l = []
    for batch_id, (q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label) in enumerate(loader):
        n_b += 1
        q = Variable(q.long())
        q_id = Variable(q_id.long())
        q_p_id = Variable(q_p_id.long())
        q_mask = Variable(q_mask.byte())
        q_o = Variable(q_o.long())
        q_o_id = Variable(q_o_id.long())
        q_o_p_id = Variable(q_o_p_id.long())
        q_o_mask = Variable(q_o_mask.byte())
        c = Variable(c.long())
        c_p_id = Variable(c_p_id.long())
        c_mask = Variable(c_mask.byte())
        label = Variable(label.long(), requires_grad=False)

        b_l = model.get_loss(q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label)
        c_l = model.get_loss_c(q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label)
        pred = model.get_tag(q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label)

        opt.zero_grad()
        b_l.backward()
        c_l.backward()
        opt.step()

        e_loss += sum(b_l.data.cpu().numpy())
        print('-------epoch: ', epoch, ' batch: ', batch_id, ' train_loss: ', b_l.data[0], 'context_loss: ', c_l.data[0])
        print('label: ', label.data.cpu().tolist())
        print('pred : ', pred)
        l_l.extend(label.data.cpu().tolist())
        p_l.extend(pred)

    e_loss = e_loss / n_b

    acc = accuracy_score(np.asarray(l_l), np.asarray(p_l))

    print('-----Epoch: ', epoch, ' Train loss: ', e_loss, ' Accuracy: ', acc)

    return e_loss, acc


def test(model, loader, id2ids, config):
    print('Test')
    f = open(config.target_file, 'wb')

    n_b = 0
    e_loss = 0
    l_l = []
    p_l = []
    for batch_id, (q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label) in enumerate(loader):
        n_b += 1
        q = Variable(q.long())
        q_id = Variable(q_id.long())
        q_p_id = Variable(q_p_id.long())
        q_mask = Variable(q_mask.byte())
        q_o = Variable(q_o.long())
        q_o_id = Variable(q_o_id.long())
        q_o_p_id = Variable(q_o_p_id.long())
        q_o_mask = Variable(q_o_mask.byte())
        c = Variable(c.long())
        c_p_id = Variable(c_p_id.long())
        c_mask = Variable(c_mask.byte())
        label = Variable(label.long(), requires_grad=False)

        b_l = model.get_loss(q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label)
        pred = model.get_tag(q, q_id, q_p_id, q_mask, q_o, q_o_id, q_o_p_id, q_o_mask, c, c_p_id, c_mask, label)

        e_loss += sum(b_l.data.cpu().numpy())
        print('------- batch: ', batch_id, ' test_loss: ', b_l.data[0])

        for q_i, q_o_i, t in zip(q_id, q_o_id, pred):
            d_1 = id2ids[q_i.data[0]]
            d_2 = id2ids[q_o_i.data[0]]
            d_3 = d_2.split('R')[1]
            d_4 = b_l.data[0]
            d_5 = 'true' if t[0] == 1 else 'false'
            s = d_1 + '	' + d_2 + '	' + d_3 + '	' + str(d_4) + '	' + str(d_5) + '\n'
            f.write(s.encode('utf8'))

        l_l.extend(label.data.cpu().tolist())
        p_l.extend(pred)

    e_loss = e_loss / n_b

    acc = accuracy_score(np.asarray(l_l), np.asarray(p_l))

    f.close()
    print('----- Train loss: ', e_loss, ' Accuracy: ', acc)
