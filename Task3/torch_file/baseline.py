# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 17-12-24 上午10:02
# @Author  : wang shen
# @web    : 
# @File    : baseline.py

import torch.nn.functional as F
from collections import Counter
from torch.autograd import Variable
import torch.nn as nn
import torch


class baseline(nn.Module):

    def __init__(self, vocab_size, config):
        super(baseline, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = config.embed_dim

        self.tag_size = config.tag_size
        self.max_length = config.max_length
        self.dropout = config.dropout
        self.q_h_size = config.q_state_size
        self.q_o_h_size = config.q_o_state_size
        self.c_h_size = config.c_state_size
        self.num_layer = config.num_layers
        #
        self.r = config.r
        self.w_dim = config.w_dim

        self.lookup = nn.Embedding(self.vocab_size, self.embedding_size)

        #
        self.q_o_size = self.embedding_size + self.q_h_size * self.r + self.c_h_size * self.r
        self.c_size = self.embedding_size + self.q_h_size * self.r

        self.q_lstm = nn.LSTM(self.embedding_size, self.q_h_size, self.num_layer, dropout=self.dropout)
        self.q_o_lstm = nn.LSTM(self.q_o_size, self.q_o_h_size // 2, self.num_layer, dropout=self.dropout, bidirectional=True)
        self.c_lstm = nn.LSTM(self.c_size, self.c_h_size // 2, self.num_layer, dropout=self.dropout, bidirectional=True)

        #
        self.q_linear = nn.Linear(self.q_h_size, self.w_dim)
        self.q_linear_r = nn.Linear(self.w_dim, self.r)
        self.q_o_linear = nn.Linear(self.q_o_h_size, self.tag_size)
        self.c_linear = nn.Linear(self.c_h_size, self.w_dim)
        self.c_linear_r = nn.Linear(self.w_dim, self.r)
        self.c_t_linear = nn.Linear(self.c_h_size * self.r, self.tag_size)

        self.c_norm = nn.BatchNorm1d(self.max_length, self.c_h_size * self.r)
        self.norm = nn.BatchNorm1d(self.max_length, self.q_o_h_size)

        self.loss = nn.NLLLoss()

    def init_hidden(self, num_layers, batch_size, hidden_size):
        h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))

        return (h0, c0)

    #
    def q_attention(self, x, x_mask):
        x_flat = x.view(-1, x.size(-1))
        s = self.q_linear(x_flat)
        s = self.q_linear_r(torch.tanh(s)).view(-1, x.size(1), self.r)

        x_m = x_mask.expand(self.r, *x_mask.size())
        x_m = x_m.transpose(0, 1).contiguous()
        x_m = x_m.transpose(1, -1).contiguous()
        s.data.masked_fill_(x_m.data, -float('inf'))

        w = F.softmax(s, 1)
        self.q_a = w

        out = (w.transpose(1, -1).contiguous()).bmm(x)

        return out

    #
    def c_attention(self, x, x_mask):
        x_flat = x.view(-1, x.size(-1))
        s = self.c_linear(x_flat)
        s = self.c_linear_r(torch.tanh(s)).view(-1, x.size(1), self.r)

        x_m = x_mask.expand(self.r, *x_mask.size())
        x_m = x_m.transpose(0, 1).contiguous()
        x_m = x_m.transpose(1, -1).contiguous()
        s.data.masked_fill_(x_m.data, -float('inf'))
        w = F.softmax(s, 1)
        self.c_a = w

        out = (w.transpose(1, -1).contiguous()).bmm(x)

        return out

    def get_pack_inputs(self, x, x_mask):
        l = x_mask.data.eq(0).long().sum(1).squeeze()
        _, id_sort = torch.sort(l, dim=0, descending=True)
        _, id_unsort = torch.sort(id_sort, dim=0)

        l = list(l[id_sort])

        x = x.index_select(0, Variable(id_sort))
        x = x.transpose(0, 1).contiguous()
        inp = nn.utils.rnn.pack_padded_sequence(x, l)

        return inp, Variable(id_unsort)

    def get_pad_outputs(self, x, x_mask, id_unsort):
        out = nn.utils.rnn.pad_packed_sequence(x)[0]

        out = out.transpose(0, 1).contiguous()
        out = out.index_select(0, id_unsort)

        if out.size(1) != x_mask.size(1):
            padding = torch.zeros(out.size(0),
                                  x_mask.size(1) - out.size(1),
                                  out.size(2)).type(out.data.type())
            out = torch.cat([out, Variable(padding)], 1)

        return out

    #
    def question_lstm(self, q, q_mask):
        batch_size = q.size()[0]
        q_embed = self.lookup(q)
        inp, id_unsort = self.get_pack_inputs(q_embed, q_mask)

        init_hidden = self.init_hidden(self.num_layer, batch_size, self.q_h_size)
        out, _ = self.q_lstm(inp, init_hidden)
        out = self.get_pad_outputs(out, q_mask, id_unsort)

        output = self.q_attention(out, q_mask)

        return output

    #
    def context_lstm(self, c, q_out, c_mask):
        batch_size = c.size()[0]
        c_embed = self.lookup(c)

        q_out_p = q_out.view(q_out.size(0), -1)
        q_output = q_out_p.expand(self.max_length, *q_out_p.size())
        q_output = q_output.transpose(0, 1).contiguous()

        inp = torch.cat([c_embed, q_output], -1)
        inp, id_unsort = self.get_pack_inputs(inp, c_mask)

        init_hidden = self.init_hidden(self.num_layer * 2, batch_size, self.c_h_size // 2)
        out, _ = self.c_lstm(inp, init_hidden)
        out = self.get_pad_outputs(out, c_mask, id_unsort)

        output = self.c_attention(out, c_mask)
        return output

    #
    def question_o_lstm(self, q_o, q_out, c_out, q_o_mask):
        batch_size = q_o.size()[0]
        q_o_embed = self.lookup(q_o)
        # print('q_o_embed: ', q_o_embed.size())

        q_out_p = q_out.view(q_out.size(0), -1)
        q_output = q_out_p.expand(self.max_length, *q_out_p.size())
        q_output = q_output.transpose(0, 1).contiguous()
        # print('q_output: ', q_output.size())

        c_out_p = c_out.view(c_out.size(0), -1)
        c_output = c_out_p.expand(self.max_length, *c_out_p.size())
        c_output = c_output.transpose(0, 1).contiguous()
        # print('c_output: ', c_output.size())

        inp = torch.cat([q_o_embed, q_output, c_output], -1)
        # print('inp: ', inp.size())
        inp, id_unsort = self.get_pack_inputs(inp, q_o_mask)

        init_hidden = self.init_hidden(self.num_layer * 2, batch_size, self.q_o_h_size // 2)
        out, _ = self.q_o_lstm(inp, init_hidden)

        out = self.get_pad_outputs(out, q_o_mask, id_unsort)

        return out

    def get_lstm(self, q, q_o, c, q_mask, q_o_mask, c_mask):
        q_output = self.question_lstm(q, q_mask)
        # print('q-output-size :', q_output.size())
        c_output = self.context_lstm(c, q_output, c_mask)
        # print('c-output-size: ', c_output.size())
        q_o_output = self.question_o_lstm(q_o, q_output, c_output, q_o_mask)
        # print('q-o-output-size: ', q_o_output.size())
        q_o_output = self.norm(q_o_output)
        # print('q-o-output-size: ', q_o_output.size())
        return q_o_output

    #
    def forward(self, q, q_o, c, q_mask, q_o_mask, c_mask):
        lstm = self.get_lstm(q, q_o, c, q_mask, q_o_mask, c_mask)

        s_list = []
        for t in lstm:
            tag = self.q_o_linear(t)
            tag_scores = F.log_softmax(tag, 1)
            s_list.append(tag_scores)

        s = torch.cat(s_list, 0).view(len(s_list), *s_list[0].size())
        return s

    # def get_tag(self, q, q_id, q_mask, q_o, q_o_id, q_o_mask, c, c_mask, label):
    #     s = self.forward(q, q_o, c, q_mask, q_o_mask, c_mask)
    #     s, t = torch.max(s, dim=-1)
    #
    #     r = []
    #     for k in t:
    #         k_list = k.data.cpu().tolist()
    #         tag = Counter(k_list).most_common(1)[0][0]
    #         r.append([tag])
    #     return r

    # only compare some 0/1
    def get_tag(self, q, q_id, q_mask, q_o, q_o_id, q_o_mask, c, c_mask, label):
        s = self.forward(q, q_o, c, q_mask, q_o_mask, c_mask)
        s, t = torch.max(s, dim=-1)

        l = q_o_mask.data.eq(0).long().sum(1).squeeze()

        r = []
        for k, n in zip(t, l):
            k_list = k.data.cpu().tolist()
            tag = Counter(k_list[:n]).most_common(1)[0][0]
            r.append([tag])
        return r

    def get_loss(self, q, q_id, q_mask, q_o, q_o_id, q_o_mask, c, c_mask, label):
        s = self.forward(q, q_o, c, q_mask, q_o_mask, c_mask)

        l_list = []
        label = label.expand(label.size(0), self.max_length)
        for t_s, t in zip(s, label):
            l = self.loss(t_s, t)
            l_list.append(l)

        batch_s = torch.mean(torch.cat(l_list, -1))
        return batch_s

    #
    def get_loss_c(self, q, q_id, q_mask, q_o, q_o_id, q_o_mask, c, c_mask, label):
        q_output = self.question_lstm(q, q_mask)
        c_output = self.context_lstm(c, q_output, c_mask)

        c_out_p = c_output.view(c_output.size(0), -1)
        c_out = c_out_p.expand(self.max_length, *c_out_p.size())
        c_out = c_out.transpose(0, 1).contiguous()
        c_out = self.c_norm(c_out)

        s_list = []
        for t in c_out:
            tag = self.c_t_linear(t)
            tag_scores = F.log_softmax(tag, 1)
            s_list.append(tag_scores)

        s = torch.cat(s_list, 0).view(len(s_list), *s_list[0].size())

        label = label.expand(label.size(0), self.max_length)
        l_list = []
        for t_s, t in zip(s, label):
            l = self.loss(t_s, t)
            l_list.append(l)

        batch_s = torch.mean(torch.cat(l_list, -1))

        return batch_s

    def att_loss(self, e):
        q_a = self.q_a.transpose(1, -1).contiguous()
        q_a_t = self.q_a
        q = q_a.bmm(q_a_t)

        c_a = self.c_a.transpose(1, -1).contiguous()
        c_a_t = self.c_a
        c = c_a.bmm(c_a_t)

        q_l = torch.dist(q, e, 1)
        c_l = torch.dist(c, e, 1)

        return q_l + c_l




