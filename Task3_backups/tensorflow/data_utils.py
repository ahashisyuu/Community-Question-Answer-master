# -*- coding:utf-8 -*-
"""
Create on 1st Dec, 2017
@Author: Zsank
"""

import nltk
import re
import random
import config
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

_PAD = '_PAD'
_UNK = '_UNK'
START_LIST = [_PAD, _UNK]

train_data_file = "SemEval2016-Task3-CQA-QL-train-part1.xml"

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


def tagger(text):
    return nltk.tag.pos_tag(text)


def load_data(data_dir, negative_sample=False):
    tree = ET.parse(data_dir)
    root = tree.getroot()
    count = 0
    result = {}

    for children in root:
        OrgQSubject = tagger(string_clean(children[0].text))
        OrgQuestion = tagger(string_clean(children[1].text))

        RelQSubject = tagger(string_clean(children[2][0][0].text))
        RelQuestion = (tagger('') if children[2][0][1].text == None else tagger(string_clean(children[2][0][1].text)))

        RelQ2OrgQ = 0 if children[2][0].attrib['RELQ_RELEVANCE2ORGQ'] == "Irrelevant" else 1

        both_good = False
        one_good = False
        count += 1
        if children.attrib['ORGQ_ID'] not in result.keys():
            result[children.attrib["ORGQ_ID"]] = []
        for i in range(1, len(children[2])):
            if children[2][i].attrib['RELC_RELEVANCE2RELQ'] == "Good" and \
                    children[2][i].attrib['RELC_RELEVANCE2ORGQ'] == "Good":
                RelCText = tagger(string_clean(children[2][i][0].text))
                result[children.attrib["ORGQ_ID"]].append([OrgQSubject, OrgQuestion,
                                                           RelQSubject, RelQuestion,
                                                           RelCText, RelQ2OrgQ])
                both_good = True
                break

        if not both_good:
            for i in range(1, len(children[2])):
                if (children[2][i].attrib['RELC_RELEVANCE2RELQ'] == "Good" or
                        children[2][i].attrib['RELC_RELEVANCE2ORGQ'] == "Good"):
                    RelCText = tagger(string_clean(children[2][i][0].text))
                    result[children.attrib["ORGQ_ID"]].append([OrgQSubject, OrgQuestion,
                                                               RelQSubject, RelQuestion,
                                                               RelCText, RelQ2OrgQ])
                    one_good = True
                    break

        if negative_sample and not both_good and not one_good:
            for i in range(1, len(children[2])):
                if ((children[2][i].attrib['RELC_RELEVANCE2RELQ'] != "Good" and
                     children[2][i].attrib['RELC_RELEVANCE2ORGQ'] != "Good")
                        and (children[2][i].attrib['RELC_RELEVANCE2RELQ'] != "Good" or
                             children[2][i].attrib['RELC_RELEVANCE2ORGQ'] != "Good")):
                    RelCText = tagger(string_clean(children[2][1][0].text))
                    result[children.attrib["ORGQ_ID"]].append([OrgQSubject, OrgQuestion,
                                                               RelQSubject, RelQuestion,
                                                               RelCText, RelQ2OrgQ])
                    break
    del tree
    print('the %s data is dict and have %s record' % (data_dir, count))
    return result


def load_all_data(dir):
    tree = ET.parse(dir)
    root = tree.getroot()
    dataset = {}
    count = 0
    for children in root:
        OrgQSubject = tagger(string_clean(children[0].text))
        OrgQuestion = tagger(string_clean(children[1].text))

        RelQSubject = tagger(string_clean(children[2][0][0].text))
        RelQuestion = (tagger('') if children[2][0][1].text == None else tagger(string_clean(children[2][0][1].text)))

        for i in range(1, len(children[2])):
            RelCText = tagger(string_clean(children[2][i][0].text))
            dataset[count] = [OrgQSubject, OrgQuestion,
                              RelQSubject, RelQuestion,
                              RelCText]
            count += 1
    return dataset


def load_vocab_pos_char_file(path):
    file_list = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                file_list.append(line.strip())
            else:
                break

    return file_list


def get_token_pos(list):
    v = []
    for i in range(len(list)):
        for j in range(len(list[i])-1):
            v += list[i][j]
    return v


def creat_vocab(args):
    if os.path.exists(args.word_vocab_file) and os.path.exists(args.pos_vocab_file) and os.path.exists(args.char_vocab_file):
        word_vocab_list = load_vocab_pos_char_file(args.word_vocab_file)
        pos_vocab_list = load_vocab_pos_char_file(args.pos_vocab_file)
        char_vocab_list = load_vocab_pos_char_file(args.char_vocab_file)

    else:
        train_data = load_data(args.train_data_file)
        valid_data = load_data(args.valid_data_file)
        test_data1 = load_data(args.test_data_file_1)
        test_data2 = load_data(args.test_data_file_1)

        word_vocab = {}
        pos_vocab = {}
        char_vocab = {}
        for k, v in train_data.items():
            for i in get_token_pos(v):
                if i[0] in word_vocab:
                    word_vocab[i[0]] += 1
                else:
                    word_vocab[i[0]] = 1

                if i[1] in pos_vocab:
                    pos_vocab[i[1]] += 1
                else:
                    pos_vocab[i[1]] = 1

                for j, char in enumerate(i[0]):
                    if char in char_vocab:
                        char_vocab[char] += 1
                    else:
                        char_vocab[char] = 1

        for k, v in valid_data.items():
            for i in get_token_pos(v):
                if i[0] in word_vocab:
                    word_vocab[i[0]] += 1
                else:
                    word_vocab[i[0]] = 1

                if i[1] in pos_vocab:
                    pos_vocab[i[1]] += 1
                else:
                    pos_vocab[i[1]] = 1

                for j, char in enumerate(i[0]):
                    if char in char_vocab:
                        char_vocab[char] += 1
                    else:
                        char_vocab[char] = 1

        for k, v in test_data1.items():
            for i in get_token_pos(v):
                if i[0] in word_vocab:
                    word_vocab[i[0]] += 1
                else:
                    word_vocab[i[0]] = 1

                if i[1] in pos_vocab:
                    pos_vocab[i[1]] += 1
                else:
                    pos_vocab[i[1]] = 1

                for j, char in enumerate(i[0]):
                    if char in char_vocab:
                        char_vocab[char] += 1
                    else:
                        char_vocab[char] = 1

        for k, v in test_data2.items():
            for i in get_token_pos(v):
                if i[0] in word_vocab:
                    word_vocab[i[0]] += 1
                else:
                    word_vocab[i[0]] = 1

                if i[1] in pos_vocab:
                    pos_vocab[i[1]] += 1
                else:
                    pos_vocab[i[1]] = 1

                for j, char in enumerate(i[0]):
                    if char in char_vocab:
                        char_vocab[char] += 1
                    else:
                        char_vocab[char] = 1

        word_vocab_list = START_LIST + sorted(word_vocab, key=word_vocab.get, reverse=True)
        pos_vocab_list = START_LIST + sorted(pos_vocab, key=pos_vocab.get, reverse=True)
        char_vocab_list = START_LIST + sorted(char_vocab, key=char_vocab.get, reverse=True)

        with open(args.word_vocab_file, 'w') as f:
            for w in word_vocab_list:
                f.write(w + '\n')

        with open(args.pos_vocab_file, "w") as f:
            for t in pos_vocab_list:
                f.write(t + "\n")

        with open(args.char_vocab_file, 'w') as f:
            for c in char_vocab_list:
                f.write(c + '\n')

    return word_vocab_list, pos_vocab_list, char_vocab_list


def get_embed_word(args):
    word2id = {}
    word2id['_PAD'] = 0
    word2id['_UNK'] = 1

    count = 2
    if args.embed_file:
        with open(args.embed_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sp = line.strip().split(' ')
                word2id[sp[0]] = idx + 2
                count += 1

    print('the embedding word size is %s ' % count)

    return word2id


def get_vocab2id(vocab):
    result = {}
    for idx, elem in enumerate(vocab):
        result[elem] = idx
    return result


def load_embedding(args, word2id):
    initial_embed = np.random.uniform(-0.01, 0.01, [len(word2id.keys()), args.embed_dim])

    with open(args.embed_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                sp = line.strip().split(' ')
                assert len(sp) == args.embed_dim + 1
                if sp[0] in word2id.keys():
                    initial_embed[word2id[sp[0]]] = [float(v) for v in sp[1:]]
            else:
                break

        initial_embed = np.array(initial_embed, np.float32)

    print("Success in loading init_word_embed.")
    return initial_embed


def connect_metadata(result):
    datasets = []
    count = 0
    for key in result.keys():
        num = len(result[key])
        if num != 10:
            count += 1
        for i in range(num - 1):
            for j in range(i+1, num):
                local = []
                reverse_local = []

                if result[key][i][5] == result[key][j][5]:
                    continue

                local.append(result[key][i][0])
                local.append(result[key][i][1])
                local.append(result[key][i][2])
                local.append(result[key][i][3])
                local.append(result[key][i][4])

                local.append(result[key][j][2])
                local.append(result[key][j][3])
                local.append(result[key][j][4])

                if result[key][i][5] == 1:
                    local.append(1)
                else:
                    local.append(0)

                reverse_local.append(result[key][j][0])
                reverse_local.append(result[key][j][1])
                reverse_local.append(result[key][j][2])
                reverse_local.append(result[key][j][3])
                reverse_local.append(result[key][j][4])

                reverse_local.append(result[key][i][2])
                reverse_local.append(result[key][i][3])
                reverse_local.append(result[key][i][4])

                if result[key][j][5] == 1:
                    reverse_local.append(1)
                else:
                    reverse_local.append(0)

                datasets.append(local)
                datasets.append(reverse_local)

    print('every data have 10 relation problem record , but %s have except' % count)
    return datasets


def sperate_token_pos(token_pos):
    token = []
    pos = []
    for t, p in token_pos:
        token.append(t)
        pos.append(p)

    return token, pos


def sperate_data(datasets):
    datasets_pos = []
    count = 0
    for i in range(len(datasets)):
        question_topic, question_topic_pos = sperate_token_pos(datasets[i][0])
        question, question_pos = sperate_token_pos(datasets[i][1])

        question_1_topic, question_1_topic_pos = sperate_token_pos(datasets[i][2])
        question_1, question_1_pos = sperate_token_pos(datasets[i][3])
        comment_1, comment_1_pos = sperate_token_pos(datasets[i][4])

        question_2_topic, question_2_topic_pos = sperate_token_pos(datasets[i][5])
        question_2, question_2_pos = sperate_token_pos(datasets[i][6])
        comment_2, comment_2_pos = sperate_token_pos(datasets[i][7])

        local = [question_topic, question_topic_pos,
                 question, question_pos,
                 question_1_topic, question_1_topic_pos,
                 question_1, question_1_pos,
                 comment_1, comment_1_pos,
                 question_2_topic, question_2_topic_pos,
                 question_2, question_2_pos,
                 comment_2, comment_2_pos,
                 datasets[i][8]]
        count += 1

        datasets_pos.append(local)

    print('the data have separated pos and have %s record, should equal the dataset size ' % count)

    return datasets_pos


def sperate_test_data(datasets_dict):
    # the test data set is dict
    datasets_pos = {}

    count_data = 0
    count_key = 0
    for key in datasets_dict.keys():
        id2data = []
        count_key += 1
        for i in range(len(datasets_dict[key])):
            count_data += 1
            question_topic, question_topic_pos = sperate_token_pos(datasets_dict[key][i][0])
            question, question_pos = sperate_token_pos(datasets_dict[key][i][1])

            question_1_topic, question_1_topic_pos = sperate_token_pos(datasets_dict[key][i][2])
            question_1, question_1_pos = sperate_token_pos(datasets_dict[key][i][3])
            comment_1, comment_1_pos = sperate_token_pos(datasets_dict[key][i][4])

            local = [question_topic, question_topic_pos,
                     question, question_pos,
                     question_1_topic, question_1_topic_pos,
                     question_1, question_1_pos,
                     comment_1, comment_1_pos,
                     datasets_dict[key][i][5]]

            id2data.append(local)

        datasets_pos[key] = id2data

    print('the test data have %s id and %s data' % (count_key, count_data))

    return datasets_pos


def padding(token, pos, word2id, pos2id, char2id, length, max_word_length):
    result_token = []
    result_pos = []
    result_char = []
    count = 0
    for i, t in enumerate(token):
        if t in word2id.keys():
            result_token.append(word2id[t])
            result_pos.append(pos2id[pos[i]])

            if len(t) < max_word_length:
                for j, char in enumerate(t):
                    if char in char2id.keys():
                        result_char.append(char2id[char])
                    else:
                        result_char.append(char2id['_UNK'])
            if len(t) < 10:
                result_char += [char2id['_UNK']] * (10 - len(t))

        else:
            count += 1
            result_token.append(word2id['_UNK'])
            result_pos.append(pos2id['_UNK'])

    l = len(result_token)
    if l < length:
        result_token = result_token + [word2id['_PAD']] * (length - l)
        result_pos = result_pos + [pos2id['_PAD']] * (length - l)
    elif l > length:
        result_token = result_token[0:length]
        result_pos = result_pos[0:length]
        l = length
    else:
        pass

    char_length = length * 10
    l_c = len(result_char)
    if l_c < char_length:
        result_char = result_char + [char2id['_PAD']] * (char_length - l_c)
    elif l_c > char_length:
        result_char = result_char[0:char_length]
    else:
        pass

    # print('num of the except word (is not in dict) is %s' % count)

    return result_token, result_pos, result_char, l, l_c


def make_pading(datasets_pos, word2id, pos2id, char2id,
                max_length, max_topic_length, max_word_length):
    train_datasets = []

    for k in range(len(datasets_pos)):
        question_topic, question_topic_pos, question_topic_char, question_topic_length, question_topic_char_length = padding(
            datasets_pos[k][0],
            datasets_pos[k][1],
            word2id, pos2id, char2id,
            max_topic_length,
            max_word_length)
        question, question_pos, question_char, question_length, question_char_length = padding(datasets_pos[k][2],
                                                                                               datasets_pos[k][3],
                                                                                               word2id, pos2id, char2id,
                                                                                               max_length,
                                                                                               max_word_length)

        question_1_topic, question_1_topic_pos, question_1_topic_char, question_1_topic_length, question_1_topic_char_length = padding(
            datasets_pos[k][4],
            datasets_pos[k][5],
            word2id, pos2id, char2id,
            max_topic_length,
            max_word_length)
        question_1, question_1_pos, question_1_char, question_1_length, question_1_char_length = padding(
            datasets_pos[k][6],
            datasets_pos[k][7],
            word2id, pos2id, char2id,
            max_length,
            max_word_length)
        comment_1, comment_1_pos, comment_1_char, comment_1_length, comment_1_char_length = padding(datasets_pos[k][8],
                                                                                                    datasets_pos[k][9],
                                                                                                    word2id, pos2id,
                                                                                                    char2id, max_length,
                                                                                                    max_word_length)

        question_2_topic, question_2_topic_pos, question_2_topic_char, question_2_topic_length, question_2_topic_char_length = padding(
            datasets_pos[k][10],
            datasets_pos[k][11],
            word2id, pos2id, char2id,
            max_topic_length,
            max_word_length)
        question_2, question_2_pos, question_2_char, question_2_length, question_2_char_length = padding(
            datasets_pos[k][12],
            datasets_pos[k][13],
            word2id, pos2id, char2id,
            max_length,
            max_word_length)
        comment_2, comment_2_pos, comment_2_char, comment_2_length, comment_2_char_length = padding(datasets_pos[k][14],
                                                                                                    datasets_pos[k][15],
                                                                                                    word2id, pos2id,
                                                                                                    char2id, max_length,
                                                                                                    max_word_length)

        local = [question_topic, question_topic_pos, question_topic_char, question_topic_length, question_topic_char_length,
                 question, question_pos, question_char, question_length, question_char_length,
                 question_1_topic, question_1_topic_pos, question_1_topic_char, question_1_topic_length, question_1_topic_char_length,
                 question_1, question_1_pos, question_1_char, question_1_length, question_1_char_length,
                 comment_1, comment_1_pos, comment_1_char, comment_1_length, comment_1_char_length,
                 question_2_topic, question_2_topic_pos, question_2_topic_char, question_2_topic_length, question_2_topic_char_length,
                 question_2, question_2_pos, question_2_char, question_2_length, question_2_char_length,
                 comment_2, comment_2_pos, comment_2_char, comment_2_length, comment_2_char_length,
                 datasets_pos[k][16]]

        train_datasets.append(local)

    return train_datasets


def make_test_padding(datasets_pos, word2id, pos2id, char2id,
                      max_length, max_topic_length, max_word_length):
    test_datasets = {}
    count_data = 0
    count_key = 0

    for key in datasets_pos.keys():
        id2data = []
        count_key += 1
        for k in range(len(datasets_pos[key])):
            count_data += 1
            question_topic, question_topic_pos, question_topic_char, question_topic_length, question_topic_char_length = padding(
                datasets_pos[key][k][0],
                datasets_pos[key][k][1],
                word2id, pos2id, char2id,
                max_topic_length, max_word_length)
            question, question_pos, question_char, question_length, question_char_length = padding(
                datasets_pos[key][k][2],
                datasets_pos[key][k][3],
                word2id, pos2id, char2id,
                max_length, max_word_length)
            question_1_topic, question_1_topic_pos, question_1_topic_char, question_1_topic_length, question_1_topic_char_length = padding(
                datasets_pos[key][k][4],
                datasets_pos[key][k][5],
                word2id, pos2id, char2id,
                max_topic_length, max_word_length)
            question_1, question_1_pos, question_1_char, question_1_length, question_1_char_length = padding(
                datasets_pos[key][k][6],
                datasets_pos[key][k][7],
                word2id, pos2id, char2id,
                max_length, max_word_length)
            comment_1, comment_1_pos, comment_1_char, comment_1_length, comment_1_char_length = padding(
                datasets_pos[key][k][8],
                datasets_pos[key][k][9],
                word2id, pos2id, char2id, max_length, max_word_length)

            local = [question_topic, question_topic_pos, question_topic_char, question_topic_length, question_topic_char_length,
                     question, question_pos, question_char, question_length, question_char_length,
                     question_1_topic, question_1_topic_pos, question_1_topic_char, question_1_topic_length,
                     question_1_topic_char_length,
                     question_1, question_1_pos, question_1_char, question_1_length, question_1_char_length,
                     comment_1, comment_1_pos, comment_1_char, comment_1_length, comment_1_char_length,
                     datasets_pos[key][k][10]]

            id2data.append(local)

        test_datasets[key] = id2data

    print('the padding test data have %s keys and %s data' % (count_key, count_data))

    return test_datasets


def get_batch(datasets, batch_indices):
    local = []
    element_num = len(datasets[0])
    assert element_num == 41
    for i in range(element_num):
        temp = []
        for j in batch_indices:
            temp.append(datasets[j][i])

        local.append(temp)

    return local


def batch_iter(train_datasets, batch_size, shuffle=True):

    all_batches = [np.array(row) for row in zip(*train_datasets)]
    size = len(train_datasets)
    indices = np.arange(size)
    result = []

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, size, batch_size):
        if start + batch_size >= size:
            break
        batch_indices = indices[start:start + batch_size]

        local = get_batch(train_datasets, batch_indices)

        result.append(local)

    return result

