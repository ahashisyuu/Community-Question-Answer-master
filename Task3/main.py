# -*- coding: utf-8 -*-

# !/usr/bin/env python
# @Time    : 17-11-9 下午9:56
# @Author  : wang shen
# @web    : 
# @File    : main.py

import time
import datetime
import numpy as np
import os
import tensorflow as tf
import Triangulation_Approach_Community_QA
import data_utils
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def MRR(out, th):
    mrr = 0.0
    for qid in out:
        candidates = out[qid]
        for i in range(min(th, len(candidates))):
            if candidates[i][1] == 1 and candidates[i][2] == 1:
                mrr += 1.0 / (i + 1)
                break

    return mrr / len(out)


def MAP(out, th):
    num_queries = len(out)
    MAP = 0.0
    for qid in out:
        candidates = out[qid]
        # compute the number of relevant docs
        # get a list of precisions in the range(0,th)
        avg_prec = 0
        precisions = []
        num_correct = 0
        for i in range(min(th, len(candidates))):
            if candidates[i][1] == 1 and candidates[i][2] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))

        if precisions:
            avg_prec = sum(precisions) / len(precisions)

        MAP += avg_prec
    return MAP / num_queries


def AvgRec(out, th):
    acc = [0.0] * th
    maxrel = [0.0] * th
    for qid in out:
        relevant = out[qid]
        num_relevant = sum([1.0 for x in relevant if x[1] == 1 and x[2] == 1])
        # print num_relevant
        for i in range(min(th, len(relevant))):
            if relevant[i][1] == 1 and relevant[i][2] == 1:
                acc[i] += 1.0
        for i in range(th):
            maxrel[i] += min(i + 1, num_relevant)
    for i in range(1, th):
        acc[i] += acc[i - 1]

    result = [a / numrel for a, numrel in zip(acc, maxrel) if numrel != 0]
    return sum(result) / len(result) if len(result) != 0 else 0


def apk(actual, predicted, k=config.get_args().batch_size):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    '''
    Let’s say, we recommended 7 products and 1st, 4th, 5th, 6th product was correct.
     so the result would look like — 1, 0, 0, 1, 1, 1, 0.
    In this case,
    The precision at 1 will be: 1/1 = 1
    The precision at 2 will be: 0
    The precision at 3 will be: 0
    The precision at 4 will be: 2/4 = 0.5
    The precision at 5 will be: 3/5 = 0.6
    The precision at 6 will be: 4/6 = 0.66
    The precision at 7 will be: 0
    Average Precision will be: 1 + 0 + 0 + 0.5 + 0.6 + 0.66 + 0 /4 = 0.69
     — Please note that here we always sum over the correct images,
     hence we are dividing by 4 and not 7.
    '''
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if predicted[i] == 1 and actual[i] == 1:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def merge_sort_part(logits, pred, label):
    # merge
    merge_list = []
    assert len(logits) == len(pred) == len(label)
    for i in range(len(logits)):
        local = []

        local.append(logits[i])
        local.append(pred[i])
        local.append(label[i])

        merge_list.append(local)

    # sort
    sort_list = sorted(merge_list, key=lambda x: (max(x[0])), reverse=True)

    # part
    pred_list = []
    label_list = []
    for i in range(len(sort_list)):
        pred_list.append(sort_list[i][1])
        label_list.append(sort_list[i][2])

    return pred_list, label_list


def main(args):
    result = data_utils.load_data(args.train_data_file)
    print('result size : %s ' % len(result))

    datasets = data_utils.connect_metadata(result)
    print('dataset size : %s ' % len(datasets))

    datasets_pos = data_utils.sperate_data(datasets)

    word_vocab_list, pos_vocab_list = data_utils.creat_vocab(args)

    args.word_vocab_size = len(word_vocab_list)
    print('word size: %s' % len(word_vocab_list))

    args.pos_vocab_size = len(pos_vocab_list)
    print('pos size: %s ' % len(pos_vocab_list))

    word2id = data_utils.get_vocab2id(word_vocab_list)
    pos2id = data_utils.get_vocab2id(pos_vocab_list)

    embed_word2id = data_utils.get_embed_word(args)
    print('embedding_word size: %s' % len(embed_word2id.keys()))

    pretrained_embedding = data_utils.load_embedding(args, word2id)

    train_datasets = data_utils.make_pading(datasets_pos, word2id, pos2id, args.max_length, args.max_topic_length)

    with tf.Session() as sess:
        model = Triangulation_Approach_Community_QA.QASystem(sess, pretrained_embedding, init_word_embed=True, args=args)

        sess.run(tf.global_variables_initializer())

        # model.load_checkpoints()

        if args.bool_train:
            print("-" * 20 + " Start Training: %s " % datetime.datetime.now() + "-" * 20)
            start_time = time.time()
            for epoch in range(args.num_epochs):
                for batch in data_utils.batch_iter(train_datasets, args.batch_size, args.max_length):
                    question_topic = batch[0]
                    question_topic_pos = batch[1]
                    question_topic_length = batch[2]
                    question = batch[3]
                    question_pos = batch[4]
                    question_length = batch[5]
                    question_1_topic = batch[6]
                    question_1_topic_pos = batch[7]
                    question_1_topic_length = batch[8]
                    question_1 = batch[9]
                    question_1_pos = batch[10]
                    question_1_length = batch[11]
                    comment_1 = batch[12]
                    comment_1_pos = batch[13]
                    comment_1_length = batch[14]
                    question_2_topic = batch[15]
                    question_2_topic_pos = batch[16]
                    question_2_topic_length = batch[17]
                    question_2 = batch[18]
                    question_2_pos = batch[19]
                    question_2_length = batch[20]
                    comment_2 = batch[21]
                    comment_2_pos = batch[22]
                    comment_2_length = batch[23]
                    label = batch[24]

                    loss, acc, global_step, pred, logits = model.train(np.array(question), np.array(question_pos),
                                                                       question_length, np.array(question_topic),
                                                                       np.array(question_1), np.array(question_1_pos),
                                                                       question_1_length, np.array(question_1_topic),
                                                                       np.array(comment_1), np.array(comment_1_pos),
                                                                       comment_1_length, np.array(question_1_topic),
                                                                       np.array(question_2), np.array(question_2_pos),
                                                                       question_2_length, np.array(question_2_topic),
                                                                       np.array(comment_2), np.array(comment_2_pos),
                                                                       comment_2_length, np.array(question_2_topic),
                                                                       label, args.keep_prob)

                    print("| Epoch: {:2d}".format(epoch),
                          "| Step: {:4d}".format(global_step),
                          "| Time: {:3d}s".format(int(time.time() - start_time)),
                          "| Train Loss: {:.4f}".format(loss),
                          "| Train Acc: {:.4f}".format(acc))
                    # 直观地观测效果
                    pred = [p for p in pred]
                    print("Label: {}".format(label))
                    print("Pred:  {}".format(pred))
                    pred, label = merge_sort_part(logits, pred, label)
                    score = apk(label, pred)
                    print('apk: {}'.format(score))
                    print('-' * 20)

                    if acc > args.beach_mark and args.model_count and epoch > 90:
                        model.save()
                        args.model_count -= 1

            print("-" * 20 + " Finished Training. %s " % datetime.datetime.now() + "-" * 20)

        ###########################################################################

        if args.bool_test:
            test_data_result = data_utils.load_data(args.test_data_file_1)
            print('test data result size : %s' % len(test_data_result))

            # test_data_datasets = data_utils.connect_metadata(test_data_result)
            # print('test data datasets size : %s ' % len(test_data_datasets))

            test_data_datasets_pos = data_utils.sperate_test_data(test_data_result)

            test_datasets = data_utils.make_test_padding(test_data_datasets_pos, word2id, pos2id, args.max_length,
                                                         args.max_topic_length)

            print("-" * 20 + " Start Testing: %s " % datetime.datetime.now() + "-" * 20)
            start_time = time.time()

            max_apk = 0
            min_apk = 1
            all_apk = []
            NEG = [[0] * args.max_length]
            NEG_topic = [[0] * args.max_topic_length]
            for key in test_datasets.keys():
                length = len(test_datasets[key])
                logits_list, pred_list, label_list = [], [], []
                for i in range(length):
                    question_topic = [test_datasets[key][i][0]]
                    question_topic_pos = [test_datasets[key][i][1]]
                    question_topic_length = [test_datasets[key][i][2]]
                    question = [test_datasets[key][i][3]]
                    question_pos = [test_datasets[key][i][4]]
                    question_length = [test_datasets[key][i][5]]
                    question_1_topic = [test_datasets[key][i][6]]
                    question_1_topic_pos = [test_datasets[key][i][7]]
                    question_1_topic_length = [test_datasets[key][i][8]]
                    question_1 = [test_datasets[key][i][9]]
                    question_1_pos = [test_datasets[key][i][10]]
                    question_1_length = [test_datasets[key][i][11]]
                    comment_1 = [test_datasets[key][i][12]]
                    comment_1_pos = [test_datasets[key][i][13]]
                    comment_1_length = [test_datasets[key][i][14]]

                    question_2_topic = NEG_topic
                    question_2_topic_pos = NEG_topic
                    question_2_topic_length = [0]
                    question_2 = NEG
                    question_2_pos = NEG
                    question_2_length = [0]
                    comment_2 = NEG
                    comment_2_pos = NEG
                    comment_2_length = [0]
                    label = [test_datasets[key][i][15]]

                    loss, acc, pred, logits = model.test(np.array(question), np.array(question_pos),
                                                         question_length, np.array(question_topic),
                                                         np.array(question_1), np.array(question_1_pos),
                                                         question_1_length, np.array(question_1_topic),
                                                         np.array(comment_1), np.array(comment_1_pos),
                                                         comment_1_length, np.array(question_1_topic),
                                                         np.array(question_2), np.array(question_2_pos),
                                                         question_2_length, np.array(question_2_topic),
                                                         np.array(comment_2), np.array(comment_2_pos),
                                                         comment_2_length, np.array(question_2_topic),
                                                         label, args.keep_prob)

                    print("| Time: {:3d}s".format(int(time.time() - start_time)),
                          "| Test Loss: {:.4f}".format(loss),
                          "| Test Acc: {:.4f}".format(acc))
                    # 直观地观测效果
                    logits_list += list(logits)
                    pred_list += [p for p in pred]
                    label_list += label
                    # print("Label: {}".format(label))
                    # print("Pred:  {}".format(pred))
                    print("-" * 20)

                pred, label = merge_sort_part(logits_list, pred_list, label_list)
                score = apk(label, pred, length)
                if score > max_apk:
                    max_apk = score
                if score < min_apk:
                    min_apk = score
                all_apk.append(score)

            print("-" * 20 + " End Testing: %s " % datetime.datetime.now() + "-" * 20)

            print('max_apk:{}'.format(max_apk),
                  'min_apk:{}'.format(min_apk),
                  'average:{:.4f}'.format(np.mean(all_apk)))


if __name__ == '__main__':
    args = config.get_args()
    main(args)
    # logits = [2, 5, 1, 0, 9]
    # pred = [1, 1, 1, 1, 0]
    # label = [0, 1, 2, 3, 0]
    # print(merge_sort_part(logits, pred, label))