# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-31 下午12:56
# @Author  : wang shen
# @web    : 
# @File    : analysis.py
import xml.etree.ElementTree as ET


file = ['SemEval2016-Task3-CQA-QL-test.xml', 'SemEval2017-task3-English-test.xml']
rele = ['SemEval2016-Task3-CQA-QL-test.xml.subtaskB.relevancy', 'SemEval2017-Task3-CQA-QL-test.xml.subtaskB.relevancy']
order_rele = ['2016-order.txt', '2017-order.txt']
result = ['SemEval2016-Task3-CQA-QL-test.txt', 'SemEval2017-Task3-CQA-QL-test.txt']


def analysis():
    for n in range(len(file)):
        print('analysis: ', rele[n], '\n')
        count = 0
        f = open(rele[n], 'r')

        tree = ET.parse(file[n])
        root = tree.getroot()

        for children, line in zip(root, f):
            l = 0 if children[2][0].attrib['RELQ_RELEVANCE2ORGQ'] == "Irrelevant" else 1
            if line.split()[4] == 'true':
                print('*')
                if l == 0:
                    count += 1
            elif line.split()[4] == 'false':
                if l == 1:
                    count += 1

        print('count:', count, '\n')


def order():
    for n in range(len(result)):
        print('order: ', result[n], '\n')
        f = open(result[n], 'r')
        id_list = []
        for line in f:
            if line.split()[0] not in id_list:
                id_list.append(line.split()[0])
        f.close()

        f = open(result[n], 'r')
        f_w = open(order_rele[n], 'w')
        temp = []
        k = 0
        id = id_list[k]

        for line in f:
            l = [line.split()[i] for i in range(5)]
            if l[0] == id and l[-1] == 'true':
                s = l[0] + '	' + l[1] + '	' + l[2] + '	' + l[3] + '	' + l[4] + '\n'
                f_w.write(s)
            elif l[0] == id and l[-1] == 'false':
                temp.append(l)
            elif l[0] != id:
                k += 1
                id = id_list[k]
                for j in range(len(temp)):
                    s = temp[j][0] + '	' + temp[j][1] + '	' + temp[j][2] + '	' + temp[j][3] + '	' + temp[j][4] + '\n'
                    f_w.write(s)
                temp = []

        assert k == len(id_list)-1

        f.close()
        f_w.close()


def add():
    for n in range(len(result)):
        print('add: ', result[n], '\n')

        f = open(result[n], 'r')
        f_w = open(order_rele[n], 'w')

        for line in f:
            l = [line.split()[i] for i in range(5)]
            if l[-1] == 'true':
                l[3] = str(float(l[3]) + 6.0)
            s = l[0] + '	' + l[1] + '	' + l[2] + '	' + l[3] + '	' + l[4] + '\n'
            f_w.write(s)

        f.close()
        f_w.close()


def compare():
    for n in range(len(result)):
        print('compare: ', result[n], '\n')

        f = open(result[n], 'r')
        f_w = open(rele[n], 'r')
        n = 0
        e = 0
        for l_1, l_2 in zip(f, f_w):
            n += 1
            if l_1.split()[4] != l_2.split()[4]:
                # print(l_1.split()[0])
                e += 1

        print('num, ------count: ', n, '------------', e)
        f.close()
        f_w.close()


if __name__ == '__main__':
    # analysis()
    # order()
    # add()
    compare()

'''
v1--v2 ---2017-12-26

v3--2017-12-29----attention

v4----17-12-30---pos
num, ------count:  700 ------------ 261
num, ------count:  880 ------------ 289

v5----18-1--1--pos---char--498
num, ------count:  700 ------------ 249
num, ------count:  880 ------------ 293
'''
