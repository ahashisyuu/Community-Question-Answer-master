# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-23 下午8:43
# @Author  : wang shen
# @web    : 
# @File    : main.py
import torch.utils.data as data
from datetime import datetime
import torch
import config
from torch_file.process import load
from torch_file.baseline import baseline
from torch_file.train import train, test
import h5py
import os


class loadDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.n = len(self.file['question'][:])

    def __getitem__(self, index):
        question = self.file['question'][index]
        q_id = self.file['q_id'][index]
        q_p_id = self.file['q_p_id'][index]
        q_mask = self.file['q_mask'][index]
        question_o = self.file['question_o'][index]
        q_o_id = self.file['q_o_id'][index]
        q_o_p_id = self.file['q_o_p_id'][index]
        q_o_mask = self.file['q_o_mask'][index]
        context = self.file['context'][index]
        c_p_id = self.file['c_p_id'][index]
        c_mask = self.file['c_mask'][index]
        label = self.file['label'][index]

        return question, q_id, q_p_id, q_mask, question_o, q_o_id, q_o_p_id, q_o_mask, context, c_p_id, c_mask, label

    def __len__(self):
        return self.n


def train_model(train_loader, val_loader, config):
    config.model_dir = '../torch_file/model_' + str(datetime.now()).split('.')[0].split()[0]
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    if os.path.exists(config.word_vocab_h5):
        word_set, word2id, vocab_size = load(config.word_vocab_h5)
    else:
        print('word vocab error')
        return

    if os.path.exists(config.pos_h5):
        pos_set, pos2id, pos_size = load(config.pos_h5)
    else:
        print('pos error')
        return

    model = baseline(vocab_size, pos_size, config)
    train(model, train_loader, config)


def test_model(test_loader, config):
    if os.path.exists(config.id_h5):
        ids_set, ids2id, ids_set = load(config.id_h5)
    else:
        print('ids error')
        return
    id2ids = dict(zip(ids2id.values(), ids2id.keys()))

    model = torch.load(config.load_model_dir)
    test(model, test_loader, id2ids, config)


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

    print('start train model')
    # train_model(train_loader, val_loader, config)
    print('end train model', '\n', '\n', '\n')

    print('start test--2016 model')
    config.target_file = config.result_data_file_2016
    test_model(test_loader_2016, config)
    print('end test--2016 model', '\n', '\n', '\n')

    print('start test--2017 model')
    config.target_file = config.result_data_file_2017
    test_model(test_loader_2017, config)
    print('end test--2017 model', '\n', '\n', '\n')
