# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-23 下午2:58
# @Author  : wang shen
# @web    : 
# @File    : process.py

import xml.etree.ElementTree as ET
from torch import Tensor
import numpy as np
import config
import pickle
import nltk
import h5py
import os
import re

_PAD = '_PAD'
_UNK = '_UNK'

TOKENIZER_RE = re.compile(r"[a-zA-Z0-9,.!?_]+")


def string_clean(string):
    string = string.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ")
    string = string.replace(";", " ").replace(":", " ").replace("(", " ").replace(")", " ")
    string = string.replace("_", " ").replace("-", " ")
    string = string.replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'m", " am")
    string = string.replace("'d", " would").replace("'ll", " will").replace("can't", "can not")
    string = string.lower().strip()

    words = re.findall(TOKENIZER_RE, string)
    words = [w for w in words if len(w) > 1 or w in {'i', 'a', ',', '.', '!', '?'}]
    return words


def load_words(path, result):
    print('load words from ', path, '\n')
    tree = ET.parse(path)
    root = tree.getroot()

    local = []
    for children in root:
        OrgQSubject = string_clean(children[0].text)
        local.extend(OrgQSubject)
        OrgQuestion = string_clean(children[1].text)
        local.extend(OrgQuestion)
        RelQSubject = string_clean(children[2][0][0].text)
        local.extend(RelQSubject)
        RelQuestion = '' if children[2][0][1].text == None else string_clean(children[2][0][1].text)
        local.extend(RelQuestion)

        for k in range(1, len(children[2])):
            RelCText = string_clean(children[2][k][0].text)
            local.extend(RelCText)

    for w in local:
        result.add(w)

    return result


def get_vocab(task_paths):
    print('Getting vocabulary')
    result = set()
    for p in task_paths:
        result = load_words(p, result)

    result = sorted(list(result))
    word_set = [_PAD]
    word_set.extend(list(result))
    word_size = len(word_set)
    word2id = dict(zip(word_set, range(word_size)))

    print('Vacabulary size:', word_size, '\n')
    return word_set, word2id, word_size

##############


def load_id(path, result):
    print('load ids from ', path, '\n')
    tree = ET.parse(path)
    root = tree.getroot()

    for children in root:
        result.add(children.attrib['ORGQ_ID'])
        result.add(children[2][0].attrib['RELQ_ID'])

    return result


def get_ids(task_paths):
    print('Getting ids')
    result = set()
    for p in task_paths:
        result = load_id(p, result)

    result = sorted(list(result))
    ids_set = list(result)
    ids_size = len(ids_set)
    ids2id = dict(zip(ids_set, range(ids_size)))

    print('Ids size:', ids_size, '\n')
    return ids_set, ids2id, ids_size

###############


def tag(text):
    p = []
    r = nltk.tag.pos_tag(text)
    for k in range(len(r)):
        p.extend(r[k][1])

    return p


def load_pos(path, result):
    print('load pos from ', path, '\n')
    tree = ET.parse(path)
    root = tree.getroot()

    l = []
    for children in root:
        OrgQSubject = tag(string_clean(children[0].text))
        l.extend(OrgQSubject)
        OrqQuestion = tag(string_clean(children[1].text))
        l.extend(OrqQuestion)
        RelQSubject = tag(string_clean(children[2][0][0].text))
        l.extend(RelQSubject)
        RelQuestion = '' if children[2][0][1].text==None else tag(string_clean(children[2][0][1].text))
        l.extend(RelQuestion)

        for k in range(1, len(children[2])):
            RelText = tag(string_clean(children[2][k][0].text))
            l.extend(RelText)

    for w in l:
        result.add(w)

    return result


def get_pos(task_paths):
    print('Getting pos')
    result = set()
    for p in task_paths:
        result = load_pos(p, result)

    result = sorted(list(result))
    pos_set = [_PAD]
    pos_set.extend(list(result))
    pos_size = len(pos_set)
    pos2id = dict(zip(pos_set, range(pos_size)))

    print('Pos size: ', pos_size, '\n')
    return pos_set, pos2id, pos_size

####################


def save(path, input2id):
    print('saving')
    f = open(path, 'wb')
    pickle.dump(input2id, f)
    f.close()


def load(path):
    print('loading ')
    f = open(path, 'rb')
    input2id = pickle.load(f)
    input_set = list(input2id.keys())
    input_size = len(input_set)
    f.close()

    print('load data size:', input_size, '\n')
    return input_set, input2id, input_size

################################################################


def pad_seq(seq, seq_size, word2id):
    vector = []

    for i in range(seq_size):
        if i >= len(seq):
            vector.append(word2id[_PAD])
        else:
            vector.append(word2id[seq[i]])
    mask = Tensor(vector).le(0).tolist()

    return vector, mask


def save_h5(old_path, new_path, word2id, ids2id, pos2id, max_length):
    print('save data from ', old_path, 'to the ', new_path)
    f = h5py.File(new_path, 'w')
    question, q_id, q_p_id, q_mask = [], [], [], []
    question_o, q_o_id, q_o_p_id, q_o_mask = [], [], [], []
    context, c_p_id, c_mask = [], [], []
    label = []

    tree = ET.parse(old_path)
    root = tree.getroot()

    q_count = 0
    c_count = 0
    for children in root:
        OrgQueId = children.attrib['ORGQ_ID']
        OrgQuestion = string_clean(children[1].text)

        if len(OrgQuestion) == 0:
            OrgQuestion = string_clean(children[0].text)

        if len(OrgQuestion) == 0:
            print('OrgQuestion', OrgQueId, 'error', '*********************************')

        q, q_m = pad_seq(OrgQuestion, max_length, word2id)
        q_p, _ = pad_seq(tag(OrgQuestion), max_length, pos2id)

        RelQueId = children[2][0].attrib['RELQ_ID']
        RelQuestion = '' if children[2][0][1].text == None else string_clean(children[2][0][1].text)

        if len(RelQuestion) == 0:
            RelQuestion = '' if children[2][0][0].text == None else string_clean(children[2][0][0].text)

        if len(RelQuestion) == 0:
            print(OrgQueId, 'RelQuestion', RelQueId, 'error', '**********************************')

        q_o, q_o_m = pad_seq(RelQuestion, max_length, word2id)
        q_o_p, _ = pad_seq(tag(RelQuestion), max_length, pos2id)

        l = 0 if children[2][0].attrib['RELQ_RELEVANCE2ORGQ'] == "Irrelevant" else 1

        q_count += 1
        RelText = []

        for i in range(1, len(children[2])):
            if l == 1:
                if children[2][i].attrib['RELC_RELEVANCE2ORGQ'] == "Good":
                    RelText = string_clean(children[2][i][0].text)
                    if len(RelText) != 0:
                        break
            if l == 0:
                if children[2][i].attrib['RELC_RELEVANCE2ORGQ'] == 'Bad':
                    RelText = string_clean(children[2][i][0].text)
                    if len(RelText) != 0:
                        break

        if len(RelText) == 0 and l == 1:
            for i in range(1, len(children[2])):
                if children[2][i].attrib['RELC_RELEVANCE2ORGQ'] == "PotentiallyUseful":
                    RelText = string_clean(children[2][i][0].text)
                    if RelText != 0:
                        break

        if len(RelText) == 0:
            RelText = RelQuestion

        if len(RelText) == 0:
            print(OrgQueId, RelQueId, 'TelText', RelText, 'error', '*********************************')

        c, c_m = pad_seq(RelText, max_length, word2id)
        c_p, _ = pad_seq(tag(RelText), max_length, pos2id)
        c_count += 1

        question.append(q)
        q_id.append(ids2id[OrgQueId])
        q_p_id.append(q_p)
        q_mask.append(q_m)
        question_o.append(q_o)
        q_o_id.append(ids2id[RelQueId])
        q_o_p_id.append(q_o_p)
        q_o_mask.append(q_o_m)
        context.append(c)
        c_p_id.append(c_p)
        c_mask.append(c_m)
        label.append([l])
        # print(label)

    f.create_dataset('question', data=np.asarray(question))
    f.create_dataset('q_id', data=np.asarray(q_id))
    f.create_dataset('q_p_id', data=np.asarray(q_p_id))
    f.create_dataset('q_mask', data=np.asarray(q_mask))
    f.create_dataset('question_o', data=np.asarray(question_o))
    f.create_dataset('q_o_id', data=np.asarray(q_o_id))
    f.create_dataset('q_o_p_id', data=np.asarray(q_o_p_id))
    f.create_dataset('q_o_mask', data=np.asarray(q_o_mask))
    f.create_dataset('context', data=np.asarray(context))
    f.create_dataset('c_p_id', data=np.asarray(c_p_id))
    f.create_dataset('c_mask', data=np.asarray(c_mask))
    f.create_dataset('label', data=np.asarray(label))

    f.close()

    print('question size :', q_count)
    print('context size :', c_count)


##################################################################


if __name__ == '__main__':
    config = config.get_args()

    task_path = [config.train_data_file, config.valid_data_file, config.test_data_file_2016, config.test_data_file_2017]
    h5py_path = [config.train_data_h5, config.valid_data_h5, config.test_data_h5_2016, config.test_data_h5_2017]

    print('load ids')
    if not os.path.exists(config.id_h5):
        ids_set, ids2id, ids_set = get_ids(task_path)
        save(config.id_h5, ids2id)
    else:
        ids_set, ids2id, ids_set = load(config.id_h5)

    print('load vocab')
    if not os.path.exists(config.word_vocab_h5):
        word_set, word2id, word_size = get_vocab(task_path)
        save(config.word_vocab_h5, word2id)
    else:
        word_set, word2id, word_size = load(config.word_vocab_h5)

    print('load pos')
    if not os.path.exists(config.pos_h5):
        pos_set, pos2id, pos_size = get_pos(task_path)
        save(config.pos_h5, pos2id)
    else:
        pos_set, pos2id, pos_size = load(config.pos_h5)

    for i in range(len(task_path)):
        print('start saving ', task_path[i])
        save_h5(task_path[i], h5py_path[i], word2id, ids2id, pos2id, config.max_length)
        print('end save ', h5py_path[i], '\n')

