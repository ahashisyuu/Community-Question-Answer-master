# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-22 下午10:02
# @Author  : wang shen
# @web    : 
# @File    : config.py

import argparse


train_data_file = "../SemEval2016-Task3-CQA-QL-train-part1.xml"
valid_data_file = "../SemEval2016-Task3-CQA-QL-dev.xml"
test_data_file_2016 = "../SemEval2016-Task3-CQA-QL-test.xml"
test_data_file_2017 = "../SemEval2017-task3-English-test.xml"
result_data_file_2016 = "../SemEval2016-Task3-CQA-QL-test-att.txt"
result_data_crf_2016 = "../SemEval2016-Task3-CQA-QL-test-crf.txt"
result_data_file_2017 = "../SemEval2017-Task3-CQA-QL-test-att.txt"
result_data_crf_2017 = "../SemEval2017-Task3-CQA-QL-test-crf.txt"

train_data_h5 = "../SemEval2016-Task3-CQA-QL-train-part1.h5"
valid_data_h5 = "../SemEval2016-Task3-CQA-QL-dev.h5"
test_data_h5_2016 = "../SemEval2016-Task3-CQA-QL-test.h5"
test_data_h5_2017 = "../SemEval2017-task3-English-test.h5"

load_model_dir = '../torch_file/model_2017-12-29/loss0.4169_acc0.999_299'

crf_train_dir = "../crf_training.h5"
crf_test_2016_dir = '../crf_test_2016.h5'
crf_test_2017_dir = '../crf_test_2017.h5'
crf_dir = "../test.crfsuite"

word_vocab_h5 = "../word_vocab.h5"
id_h5 = '../id.h5'

word_vocab_file = "../word_vocab.txt"
pos_vocab_file = "../pos_vocab.txt"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--valid_data_file', type=str, default=valid_data_file)
    parser.add_argument('--test_data_file_2016', type=str, default=test_data_file_2016)
    parser.add_argument('--test_data_file_2017', type=str, default=test_data_file_2017)
    parser.add_argument('--result_data_file_2016', type=str, default=result_data_file_2016)
    parser.add_argument('--result_data_crf_2016', type=str, default=result_data_crf_2016)
    parser.add_argument('--result_data_file_2017', type=str, default=result_data_file_2017)
    parser.add_argument('--result_data_crf_2017', type=str, default=result_data_crf_2017)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--train_data_h5', type=str, default=train_data_h5)
    parser.add_argument('--valid_data_h5', type=str, default=valid_data_h5)
    parser.add_argument('--test_data_h5_2016', type=str, default=test_data_h5_2016)
    parser.add_argument('--test_data_h5_2017', type=str, default=test_data_h5_2017)
    parser.add_argument('--crf_train_dir', type=str, default=crf_train_dir)
    parser.add_argument('--crf_test_2016_dir', type=str, default=crf_test_2016_dir)
    parser.add_argument('--crf_test_2017_dir', type=str, default=crf_test_2017_dir)
    parser.add_argument('--crf_dir', type=str, default=crf_dir)

    parser.add_argument('--pos_vocab_file', type=str, default=pos_vocab_file)
    parser.add_argument('--word_vocab_file', type=str, default=word_vocab_file)
    parser.add_argument('--word_vocab_h5', type=str, default=word_vocab_h5)
    parser.add_argument('--id_h5', type=str, default=id_h5)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--load_model_dir', type=str, default=load_model_dir)

    parser.add_argument('--word_vocab_size', type=int, default=None)
    parser.add_argument('--pos_vocab_size', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--q_state_size', type=int, default=64)
    parser.add_argument('--q_o_state_size', type=int, default=96)
    parser.add_argument('--c_state_size', type=int, default=96)
    parser.add_argument('--tag_size', type=int, default=2)
    parser.add_argument('--w_dim', type=int, default=64)
    parser.add_argument('--r', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--max_topic_length', type=int, default=9)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--eval_iter', type=int, default=100)

    return parser.parse_args()


