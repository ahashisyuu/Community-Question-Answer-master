# -*- coding: utf-8 -*-

# !/usr/bin/env python
# @Time    : 17-12-1 下午7:25
# @Author  : wang shen
# @web    : 
# @File    : Triangulation_Approach_Community_QA.py

import tensorflow as tf
import os


class QASystem(object):
    def __init__(self, sess, pretrained_embedding, pretrained_char_embedding, init_word_embed, args):
        self.sess = sess
        self.init_word_embed = init_word_embed
        self.pretrained_embedding = pretrained_embedding
        self.pretrained_char_embedding = pretrained_char_embedding
        self.args = args
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # ==== set up placeholder tokens ====
        self.question = tf.placeholder(tf.int32, [None, None], name="q")
        self.question_pos = tf.placeholder(tf.int32, [None, None], name="q_pos")
        self.question_char = tf.placeholder(tf.int32, [None, None], name='q_char')
        self.question_topic = tf.placeholder(tf.float32, [None, None], name="q_topic")
        self.question_length = tf.placeholder(tf.int32, [None], name="q_length")
        self.question_char_length = tf.placeholder(tf.int32, [None], name='q_char_length')

        self.question_1 = tf.placeholder(tf.int32, [None, None], name="q1")
        self.question_pos_1 = tf.placeholder(tf.int32, [None, None], name="q1_pos")
        self.question_char_1 = tf.placeholder(tf.int32, [None, None], name='q1_char')
        self.question_topic_1 = tf.placeholder(tf.float32, [None, None], name="q1_topic")
        self.question_length_1 = tf.placeholder(tf.int32, [None], name="q1_length")
        self.question_char_length_1 = tf.placeholder(tf.int32, [None], name='q1_char_length')

        self.comment_1 = tf.placeholder(tf.int32, [None, None], name="c1", )
        self.comment_pos_1 = tf.placeholder(tf.int32, [None, None], name="c1_pos")
        self.comment_char_1 = tf.placeholder(tf.int32, [None, None], name='c1_char')
        self.comment_topic_1 = tf.placeholder(tf.float32, [None, None], name="c1_topic")
        self.comment_length_1 = tf.placeholder(tf.int32, [None], name="c1_length")
        self.comment_char_length_1 = tf.placeholder(tf.int32, [None], name='c1_char_length')

        self.question_2 = tf.placeholder(tf.int32, [None, None], name="q2")
        self.question_pos_2 = tf.placeholder(tf.int32, [None, None], name="q2_pos")
        self.question_char_2 = tf.placeholder(tf.int32, [None, None], name='q2_char')
        self.question_topic_2 = tf.placeholder(tf.float32, [None, None], name="q2_topic")
        self.question_length_2 = tf.placeholder(tf.int32, [None], name="q2_length")
        self.question_char_length_2 = tf.placeholder(tf.int32, [None], name='q2_char_length')

        self.comment_2 = tf.placeholder(tf.int32, [None, None], name="c2")
        self.comment_pos_2 = tf.placeholder(tf.int32, [None, None], name="c2_pos")
        self.comment_char_2 = tf.placeholder(tf.int32, [None, None], name='c2_char')
        self.comment_topic_2 = tf.placeholder(tf.float32, [None, None], name="c2_topic")
        self.comment_length_2 = tf.placeholder(tf.int32, [None], name="c2_length")
        self.comment_char_length_2 = tf.placeholder(tf.int32, [None], name='c2_char_length')

        self.labels = tf.placeholder(tf.int32, [None], name="label")
        self.keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")

        with tf.variable_scope('qa'):
            # embedding #
            self.question_embedding, self.question_1_embedding, self.comment_1_embedding, self.question_2_embedding, self.comment_2_embedding,  = self.setup_embedding()

            # system #

            self.logits, self.pred = self.setup_system()

            # loss #
            self.loss, self.acc = self.setup_loss()

            # train #
            # self.train_op = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(self.loss,
            #                                                                                     self.global_step)
            self.train_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss, self.global_step)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

    def setup_embedding(self):
        with tf.variable_scope('embedding'):
            if not self.init_word_embed:
                self.pretrained_embedding = tf.truncated_normal([self.args.word_vocab_size, self.args.embed_dim],
                                                                stddev=0.5)
                self.pretrained_char_embedding = tf.truncated_normal([self.args.char_vocab_size, self.args.embed_dim],
                                                                     stddev=0.1)

            question_embed = tf.nn.embedding_lookup(self.pretrained_embedding, self.question)
            question1_embed = tf.nn.embedding_lookup(self.pretrained_embedding, self.question_1)
            comment1_embed = tf.nn.embedding_lookup(self.pretrained_embedding, self.comment_1)
            question2_embed = tf.nn.embedding_lookup(self.pretrained_embedding, self.question_2)
            comment2_embed = tf.nn.embedding_lookup(self.pretrained_embedding, self.comment_2)

            #
            # question_char_embed = tf.nn.embedding_lookup(self.pretrained_char_embedding, self.question_char)
            # question1_char_embed = tf.nn.embedding_lookup(self.pretrained_char_embedding, self.question_char_1)
            # comment1_char_embed = tf.nn.embedding_lookup(self.pretrained_char_embedding, self.comment_char_1)
            # question2_char_embed = tf.nn.embedding_lookup(self.pretrained_char_embedding, self.question_char_2)
            # comment2_char_embed = tf.nn.embedding_lookup(self.pretrained_char_embedding, self.comment_char_2)
            #

            init_pos_embed = tf.truncated_normal([self.args.pos_vocab_size, self.args.embed_dim], stddev=0.5)
            self.pos_embedding = tf.Variable(init_pos_embed, name="pos_embedding", trainable=True)
            question_pos_embed = tf.nn.embedding_lookup(self.pos_embedding, self.question_pos)
            question1_pos_embed = tf.nn.embedding_lookup(self.pos_embedding, self.question_pos_1)
            comment1_pos_embed = tf.nn.embedding_lookup(self.pos_embedding, self.comment_pos_1)
            question2_pos_embed = tf.nn.embedding_lookup(self.pos_embedding, self.question_pos_2)
            comment2_pos_embed = tf.nn.embedding_lookup(self.pos_embedding, self.comment_pos_2)

            question = question_embed + question_pos_embed
            question1 = question1_embed + question1_pos_embed
            comment1 = comment1_embed + comment1_pos_embed
            question2 = question2_embed + question2_pos_embed
            comment2 = comment2_embed + comment2_pos_embed

            return question, question1, comment1, question2, comment2

        # return question, question_char_embed, question1, question1_char_embed, comment1, comment1_char_embed, question2, question2_char_embed, comment2, comment2_char_embed

    def rnn_encoder(self, input, input_length=None, name=None, keep_prob=1.0):
        with tf.variable_scope('{}_RNN'.format(name)):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.state_size, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.state_size, state_is_tuple=True)

            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob)

            (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=input,
                sequence_length=input_length,
                dtype=tf.float32
            )
            outputs = tf.reshape(tf.concat([outputs_fw, outputs_bw], axis=2),
                                 [-1, self.args.max_length])
            final_state = tf.concat([final_state_fw[1], final_state_bw[1]], axis=1)
            final_state = tf.reshape(final_state, [-1, self.args.max_length, 2 * self.args.state_size])
            W = tf.Variable(tf.truncated_normal([self.args.max_length, 1], stddev=0.1, name='{}_W'.format(name)))
            out = tf.reshape(tf.matmul(outputs, W), [-1, 2 * self.args.state_size])
            return out

    def attention(self, original_question, target_question, name='question'):
        with tf.variable_scope('{}_attention'.format(name)):
            W = tf.get_variable('W', shape=[2 * self.args.state_size, 2 * self.args.state_size])
            original_question = tf.reshape(original_question, [-1, 2 * self.args.state_size])
            target_question = tf.reshape(target_question, [-1, 2 * self.args.state_size])

            similarty = tf.matmul(tf.matmul(original_question, W), tf.transpose(target_question, perm=[1, 0]))
            a = tf.nn.softmax(similarty)

            attention_q = tf.matmul(a, target_question)
            attention_q = tf.reshape(attention_q, [-1, 2 * self.args.state_size])

            return attention_q

    def cnn_encoder(self, input, input_length, operation='word', name=None, keep_prob=1.0):
        max_length = self.args.max_length if operation == 'word' else self.args.max_length * 10
        # can assert
        re_input = tf.reshape(input, [self.args.batch_size, max_length, self.args.embed_dim, 1])
        with tf.variable_scope("{}_CNN".format(name)):
            pooled_outputs = []
            for i, filter_size in enumerate(self.args.filter_sizes):
                with tf.variable_scope("{}_filter_{}".format(name, filter_size)):
                    # convolution + pool
                    filter_shape = [filter_size, self.args.embed_dim, 1, self.args.num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, name="W"))
                    b = tf.Variable(tf.constant(0.1, shape=[self.args.num_filter]), name="b")

                    conv = tf.nn.relu(
                        tf.nn.bias_add(tf.nn.conv2d(re_input, W, strides=[1, 1, 1, 1], padding="VALID"), b))

                    pooled = tf.nn.max_pool(conv, ksize=[1, max_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding="VALID")
                    pooled_outputs.append(pooled)

                    # # convolution + pool + convolution + pool
                    # filter_shape_1 = [filter_size, self.args.embed_dim, 1, self.args.convolution_out_channel_1]
                    # W_1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1, name='{}_W_1'.format(name)))
                    # b_1 = tf.Variable(tf.constant(0.1, shape=[self.args.convolution_out_channel_1]),
                    #                   name='{}_b_1'.format(name))
                    # conv_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(re_input, W_1,
                    #                                                 strides=[1, 1, 1, 1], padding='VALID'), b_1))
                    # pool_1 = tf.nn.max_pool(conv_1, ksize=[1, filter_size, 1, 1],
                    #                         strides=[1, 1, 1, 1], padding='VALID')
                    #
                    # filter_shape_2 = [filter_size, 1, self.args.convolution_out_channel_1, self.args.num_filter]
                    # W_2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1, name='{}_W_2'.format(name)))
                    # b_2 = tf.Variable(tf.constant(0.1, shape=[self.args.num_filter]), name='{}_b_2'.format(name))
                    # conv_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool_1, W_2,
                    #                                                 strides=[1, 1, 1, 1], padding='VALID'), b_2))
                    # pool_2 = tf.nn.max_pool(conv_2, ksize=[1, self.args.max_length - 3 * filter_size + 3, 1, 1],
                    #                         strides=[1, 1, 1, 1], padding='VALID')
                    #
                    # pooled_outputs.append(pool_2)

            # (B, 1, 1, 3*state_size)
            total_channels = len(self.args.filter_sizes) * self.args.num_filter
            pool_output = tf.reshape(tf.concat(pooled_outputs, 3), [-1, total_channels])

            # dropout #
            output = tf.nn.dropout(pool_output, keep_prob)

            _w = tf.Variable(tf.truncated_normal([total_channels, 2 * self.args.state_size], stddev=0.1,
                                                 name='{}__w'.format(name)))
            state = tf.reshape(tf.matmul(output, _w), [-1, 2 * self.args.state_size])

            return state

    def hq_full_connect(self, hq, num_state, name=None):
        with tf.variable_scope('{}_full_connect'.format(name)):
            hq = tf.reshape(hq, [-1, num_state * self.args.state_size])
            W = tf.get_variable('w', shape=[num_state * self.args.state_size, 1])
            b = tf.get_variable('b', shape=[1])

            output = tf.tanh(tf.matmul(hq, W) + b)
            # output = tf.reshape(output, [-1])
            return output

    def setup_system(self):
        if self.args.encoder_mode == 'rnn':
            self.encoder = self.rnn_encoder
        else:
            self.encoder = self.cnn_encoder

        question_state = self.encoder(self.question_embedding, self.question_length, name="q_encoder", keep_prob=self.keep_prob)
        question_state_1 = self.encoder(self.question_1_embedding, self.question_length_1, name="q1_encoder", keep_prob=self.keep_prob)
        comment_state_1 = self.encoder(self.comment_1_embedding, self.comment_length_1, name='c1_encoder', keep_prob=self.keep_prob)
        question_state_2 = self.encoder(self.question_2_embedding, self.question_length_2, name='q2_encoder', keep_prob=self.keep_prob)
        comment_state_2 = self.encoder(self.comment_2_embedding, self.comment_length_2, name='c2_encoder', keep_prob=self.keep_prob)

        if self.args.attention:
            question_state_1 = self.attention(question_state, question_state_1, name='q_1')
            question_state_2 = self.attention(question_state, question_state_2, name='q_2')
            comment_state_1 = self.attention(question_state, comment_state_1, name='c_1')
            comment_state_2 = self.attention(question_state, comment_state_2, name='c_2')

        # hq_1, hq_2, hq_3 #
        hq_1 = tf.concat([question_state, question_state_1, question_state_2], axis=1)
        hq_1_output = self.hq_full_connect(hq_1, 2 * 3, name='hq_1')
        hq_2 = tf.concat([question_state, comment_state_1, comment_state_2], axis=1)
        hq_2_output = self.hq_full_connect(hq_2, 2 * 3, name='hq_2')
        hq_4 = tf.concat([question_state, question_state_1, comment_state_1], axis=1)
        hq_4_output = self.hq_full_connect(hq_4, 2 * 3, name='hq_4')
        hq_5 = tf.concat([question_state, question_state_2, comment_state_2], axis=1)
        hq_5_output = self.hq_full_connect(hq_5, 2 * 3, name='hq_5')

        hq_3 = tf.concat([question_state_1, comment_state_1, question_state_2, comment_state_2], axis=1)
        hq_3_output = self.hq_full_connect(hq_3, 2 * 4, name='hq_3')

        # concat #
        output = tf.concat([hq_1_output, hq_2_output, hq_3_output, hq_4_output, hq_5_output], axis=1)
        W = tf.get_variable('W_concat', [5, 2])
        b = tf.get_variable('b_concat', [2])
        logits = tf.reshape(tf.matmul(output, W) + b, [-1, 2])
        pred = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

        # probs = tf.nn.sigmoid(logits)
        # zero_label = tf.zeros_like(probs, tf.int32)
        # one_label = tf.ones_like(probs, tf.int32)
        # prob_threshold = tf.ones_like(probs, tf.float32) * 0.5
        # prob_mask = tf.greater(probs, prob_threshold)
        # pred = tf.where(prob_mask, one_label, zero_label)

        return logits, pred

    def setup_loss(self):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.labels, tf.float32)))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
        return loss, accuracy

    def train(self, q, q_pos, q_char, q_topic, q_l, q_char_l,
              q1, q1_pos, q1_char, q1_topic, q1_l, q1_char_l,
              c1, c1_pos, c1_char, c1_topic, c1_l, c1_char_l,
              q2, q2_pos, q2_char, q2_topic, q2_l, q2_char_l,
              c2, c2_pos, c2_char, c2_topic, c2_l, c2_char_l,
              label, keep_prob):
        feed_dict = {self.question: q, self.question_pos: q_pos, self.question_char: q_char,
                     self.question_length: q_l,
                     self.question_topic: q_topic, self.question_char_length: q_char_l,
                     self.question_1: q1, self.question_pos_1: q1_pos, self.question_char_1: q1_char,
                     self.question_length_1: q1_l,
                     self.question_topic_1: q1_topic, self.question_char_length_1: q1_char_l,
                     self.comment_1: c1, self.comment_pos_1: c1_pos, self.comment_char_1: c1_char,
                     self.comment_length_1: c1_l,
                     self.comment_topic_1: c1_topic, self.comment_char_length_1: c1_char_l,
                     self.question_2: q2, self.question_pos_2: q2_pos, self.question_char_2: q2_char,
                     self.question_length_2: q2_l,
                     self.question_topic_2: q2_topic, self.question_char_length_2: q2_char_l,
                     self.comment_2: c2, self.comment_pos_2: c2_pos, self.comment_char_2: c2_char,
                     self.comment_length_2: c2_l,
                     self.comment_topic_2: c2_topic, self.comment_char_length_2: c2_char_l,
                     self.labels: label, self.keep_prob: keep_prob}
        _, loss, acc, global_step, pred, logits = self.sess.run(
            [self.train_op, self.loss, self.acc, self.global_step, self.pred, self.logits],
            feed_dict=feed_dict)
        return loss, acc, global_step, pred, logits

    def test(self, q, q_pos, q_char, q_topic, q_l, q_char_l,
              q1, q1_pos, q1_char, q1_topic, q1_l, q1_char_l,
              c1, c1_pos, c1_char, c1_topic, c1_l, c1_char_l,
              q2, q2_pos, q2_char, q2_topic, q2_l, q2_char_l,
              c2, c2_pos, c2_char, c2_topic, c2_l, c2_char_l,
              label, keep_prob):

        feed_dict = {self.question: q, self.question_pos: q_pos, self.question_char: q_char,
                     self.question_length: q_l,
                     self.question_topic: q_topic, self.question_char_length: q_char_l,
                     self.question_1: q1, self.question_pos_1: q1_pos, self.question_char_1: q1_char,
                     self.question_length_1: q1_l,
                     self.question_topic_1: q1_topic, self.question_char_length_1: q1_char_l,
                     self.comment_1: c1, self.comment_pos_1: c1_pos, self.comment_char_1: c1_char,
                     self.comment_length_1: c1_l,
                     self.comment_topic_1: c1_topic, self.comment_char_length_1: c1_char_l,
                     self.question_2: q2, self.question_pos_2: q2_pos, self.question_char_2: q2_char,
                     self.question_length_2: q2_l,
                     self.question_topic_2: q2_topic, self.question_char_length_2: q2_char_l,
                     self.comment_2: c2, self.comment_pos_2: c2_pos, self.comment_char_2: c2_char,
                     self.comment_length_2: c2_l,
                     self.comment_topic_2: c2_topic, self.comment_char_length_2: c2_char_l,
                     self.labels: label, self.keep_prob: keep_prob}
        loss, acc, pred, logits = self.sess.run([self.loss, self.acc, self.pred, self.logits], feed_dict=feed_dict)
        return loss, acc, pred, logits

    def save(self):
        print("*** Saving checkpoints_1 ***")
        if not os.path.exists(self.args.checkpoint_dir):
            os.mkdir(self.args.checkpoint_dir)
        fpath = os.path.join(self.args.checkpoint_dir, "{}.ckpt".format(self.__class__.__name__))
        save_path = self.saver.save(self.sess, fpath, global_step=self.global_step)
        print("$ Model saved in file: %s" % save_path)

    def load_checkpoints(self, global_step=None):
        print("*** Loading Checkpoints ***")
        ckpt = tf.train.get_checkpoint_state(self.args.checkpoint_dir)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            ckpt_name = ckpt.model_checkpoint_path
            if global_step:
                ckpt_name = "{}-{}".format(ckpt_name.split("-")[0], str(global_step))
            print("*** Loding path: {} ***".format(ckpt_name))
            self.saver.restore(self.sess, ckpt_name)
            print("*** load model success ****")
            return True
        else:
            print("!!! Load failed !!!")
            return False
