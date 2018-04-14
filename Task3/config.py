# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-11-9 下午8:05
# @Author  : wang shen
# @web    : 
# @File    : config.py

import argparse
#
# log_dir = r"E:/codes/Task3/logs"
# checkpoint_dir = r"E:/codes/Task3/checkpoints"

checkpoint_dir = "checkpoints"
log_dir = 'logs'

train_data_file = "SemEval2016-Task3-CQA-QL-train-part1.xml"
valid_data_file = "SemEval2016-Task3-CQA-QL-dev.xml"
test_data_file_1 = "SemEval2016-Task3-CQA-QL-test.xml"
test_data_file_2 = "SemEval2017-task3-English-test.xml"

stopword_file = "stopword.txt"
embed_file = "glove.6B.100d.txt"
word_vocab_file = "word_vocab.txt"
pos_vocab_file = "pos_vocab.txt"


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Basics
    parser.add_argument('--debug',
                        type='bool',
                        default=False,
                        help='whether it is debug mode')

    # Data file
    parser.add_argument('--train_data_file',
                        type=str,
                        default=train_data_file)
    parser.add_argument('--valid_data_file',
                        type=str,
                        default=valid_data_file)
    parser.add_argument('--test_data_file_1',
                        type=str,
                        default=test_data_file_1)
    parser.add_argument('--test_data_file_2',
                        type=str,
                        default=test_data_file_2)
    parser.add_argument('--stopword_file',
                        type=str,
                        default=stopword_file)
    parser.add_argument('--pos_vocab_file',
                        type=str,
                        default=pos_vocab_file)
    parser.add_argument('--embed_file',
                        type=str,
                        default=embed_file)
    parser.add_argument('--word_vocab_file',
                        type=str,
                        default=word_vocab_file)
    # dir
    parser.add_argument('--log_dir',
                        type=str,
                        default=log_dir,
                        help='log_dir')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default=checkpoint_dir,
                        help='checkpoint_dir')

    # Model details
    parser.add_argument('--word_vocab_size',
                        type=int,
                        default=None,
                        help='Word_vocab_size')
    parser.add_argument('--pos_vocab_size',
                        type=int,
                        default=None,
                        help='Pos_vocab_size')
    parser.add_argument('--embed_dim',
                        type=int,
                        default=100,
                        help='Default embedding size')
    parser.add_argument('--state_size',
                        type=int,
                        default=64,
                        help='Hidden size of NN units')
    parser.add_argument('--bool_train',
                        type=bool,
                        default=True,
                        help='it is to train')
    parser.add_argument('--bool_test',
                        type=bool,
                        default=True,
                        help='it is to test')
    parser.add_argument('--filter_sizes',
                        type=list,
                        default=[3, 4, 5],
                        help='filter_sizes for CNN')
    parser.add_argument('--num_filter',
                        type=int,
                        default=100,
                        help='num_filter')
    parser.add_argument('--convolution_out_channel_1',
                        type=int,
                        default=50,
                        help='it must  be < the num filter')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('--batch_test_size',
                        type=int,
                        default=1,
                        help='batch test size')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.0001,
                        help='Learn rate for SGD')
    parser.add_argument('--keep_prob',
                        type=float,
                        default=0.8,
                        help='Dropout rate')
    parser.add_argument('--max_length',
                        type=int,
                        default=150,
                        help='Max length of input')
    parser.add_argument('--max_topic_length',
                        type=int,
                        default=9,
                        help='max length of topic')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='num of epochs')
    parser.add_argument('--eval_iter',
                        type=int,
                        default=100,
                        help='Evaluation on dev set after K updates')
    parser.add_argument('--encoder_mode',
                        type=str,
                        default="rnn",
                        help='Encoder mode')
    parser.add_argument('--model_count',
                        type=int,
                        default=20,
                        help='the num of model want to save')
    parser.add_argument('--beach_mark',
                        type=int,
                        default=0.99999999999,
                        help='this is base line to save moidel')

    return parser.parse_args()


test = [
    [
        [1, 2, 3], [4, 5, 6], [2, 6, 7, 8, 0], 1
    ],
    [
        [10, 20, 30], [40, 50, 60], [20, 60, 70, 80, 00], 1
    ],
    [
        [100, 2000, 300], [400, 500, 600], [200, 600, 700, 800, 000], 0
    ]
]

