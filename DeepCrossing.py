#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/6/2018 3:47 PM
# @Author  : Jianjin Zhang
# @File    : DeepCrossing.py
import math

import tensorflow as tf
import tensorflow.contrib.microsoft as mstf
from ResNetLayer import ResNetLayer
from tensorflow.python.ops import control_flow_ops


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""
    attention = tf.matmul(input_1, tf.transpose(input_2, perm=[0, 2, 1]))
    w_att_1 = tf.nn.softmax(attention, dim=1)
    w_att_2 = tf.transpose(tf.nn.softmax(attention, dim=2), perm=[0, 2, 1])

    in1_aligned = tf.matmul(tf.transpose(w_att_1, perm=[0, 2, 1]), input_1)
    in2_aligned = tf.matmul(tf.transpose(w_att_2, perm=[0, 2, 1]), input_2)

    def submult(input_1, input_2):
        """Get multiplication and subtraction then concatenate results"""
        mult = tf.multiply(input_1, input_2)
        sub = tf.subtract(input_1, input_2)
        out_ = tf.concat([sub, mult], axis=1)
        return out_

    out1 = tf.concat([input_1, in2_aligned, submult(input_1, in2_aligned)], axis=1)
    out2 = tf.concat([input_2, in1_aligned, submult(input_2, in1_aligned)], axis=1)
    return out1, out2


class DeepCrossing(mstf.Model):
    def __init__(self, data_types, data_embedding, input_names, dims, att_dim, embedding_dim, res_dims, dict_path,
                 summaries_histogram, pos_weight, deep_dims, cross_layers, double_cross, negative_count):
        super().__init__()
        self.num_data = len(data_types)
        self.data_embedding = data_embedding
        self.data_types = data_types
        self.embedding_dim = embedding_dim
        self.att_dim = att_dim
        self.num_res_layers = len(res_dims)
        self.res_dims = res_dims
        self.input_names = input_names
        self.dict_path = dict_path
        self.dims = dims
        self.summaries_histogram = summaries_histogram
        self.pos_weight = pos_weight
        self.extend_op = dict()
        self.deep_layers = len(deep_dims)
        self.deep_dims = deep_dims
        if self.deep_dims[0] == 0:
            self.deep_layers = 0
            self.deep_dims = []
        self.cross_layers = cross_layers
        self.double_cross = double_cross
        self.negative_count = negative_count

        def w_initializer(dim_in, dim_out):
            random_range = math.sqrt(6.0 / (dim_in + dim_out))
            return tf.random_uniform_initializer(-random_range, random_range)

        self.initializer = w_initializer

    def _gen_negative(self, batch_size, input_str):
        idx = []
        ido = []
        for i in range(self.negative_count + 1):
            idx.append(tf.concat([tf.range(batch_size - i, batch_size), tf.range(0, batch_size - i)], axis=0))
            ido.append(tf.range(batch_size))
        idx = tf.concat(idx, axis=0)
        ido = tf.concat(ido, axis=0)
        shuffle_order = tf.range(tf.shape(idx)[0])
        shuffle_order = tf.random_shuffle(shuffle_order)
        idx = tf.gather(idx, shuffle_order)
        ido = tf.gather(ido, shuffle_order)

        label = tf.gather(tf.concat([input_str[0], tf.zeros([self.negative_count * batch_size])], axis=0),
                          shuffle_order)
        n_features = [label, tf.gather(input_str[1], idx, axis=0)]
        for j in range(2, len(input_str)):
            n_features.append(tf.gather(tf.concat([input_str[j]] * (self.negative_count + 1), axis=0), ido, axis=0))
        return n_features

    def add_ops(self, ops):
        self.extend_op.update(ops)

    def build_graph(self):
        is_training_input = tf.placeholder(tf.bool, shape=[None], name="is_training")
        is_training = tf.reduce_all(is_training_input)
        input = []

        for i in range(len(self.data_types)):
            if self.data_types[i] == 'string' or self.data_types[i] == 'ice' or self.data_types[i] == 'web':
                input.append(tf.placeholder(tf.string, shape=[None], name=self.input_names[i]))
            if self.data_types[i] == 'float':
                input.append(tf.placeholder(tf.float32, shape=[None], name=self.input_names[i]))

        if self.negative_count > 0:

            def do_nothing(batch_size, input_str):
                return input_str

            input_tensors = tf.cond(is_training, lambda: self._gen_negative(tf.shape(input[0])[0], input),
                                    lambda: do_nothing(tf.shape(input[0])[0], input))
        else:
            input_tensors = input

        net = []
        dict_handle = mstf.dssm_dict(self.dict_path)
        all_dim = 0
        debug = []
        str_reuse = False
        ice_reuse = False
        web_reuse = False
        att_str_reuse = False
        att_ice_reuse = False
        att_web_reuse = False

        with tf.name_scope("embedding"):
            for i in range(len(self.data_embedding)):
                input_tensor = input_tensors[i + 1]
                embedding_reuse = False
                if self.data_types[i + 1] == 'string':
                    if self.data_embedding[i] == 2:
                        embedding_reuse = att_str_reuse
                        indices, ids, values, offsets = mstf.dssm_xletter(input=input_tensor, win_size=1,
                                                                          max_term_count=12,
                                                                          merge_excess=True, dict_handle=dict_handle)
                    else:
                        embedding_reuse = str_reuse
                        indices, ids, values, offsets = mstf.dssm_xletter(input=input_tensor, win_size=1,
                                                                          max_term_count=1,
                                                                          merge_excess=True, dict_handle=dict_handle)
                    offsets_to_dense = tf.segment_sum(tf.ones_like(offsets), offsets)
                    batch_id = tf.cumsum(offsets_to_dense[:-1])

                    tmp_shape = tf.cast(
                        tf.concat([tf.cast([tf.reduce_sum(offsets_to_dense)], tf.int32),
                                   tf.reduce_max(indices, keep_dims=True) + 1,
                                   tf.fill([1], self.dims[i + 1])], axis=0), tf.int64)
                    tmp_index = tf.stack([batch_id, indices, ids], axis=-1)

                    input_tensor = tf.SparseTensor(indices=tf.cast(tmp_index, tf.int64),
                                                   values=tf.ones_like(batch_id, tf.float32),
                                                   dense_shape=tmp_shape)
                if self.data_types[i + 1] == 'ice' or self.data_types[i + 1] == 'web':
                    if self.data_types[i + 1] == 'ice':
                        if self.data_embedding[i] == 2:
                            embedding_reuse = att_ice_reuse
                        else:
                            embedding_reuse = ice_reuse
                        ices = tf.string_split(input_tensor, delimiter=',')
                    else:
                        if self.data_embedding[i] == 2:
                            embedding_reuse = att_web_reuse
                        else:
                            embedding_reuse = web_reuse
                        ices = tf.string_split(input_tensor, delimiter=' ')
                    ice_indices = tf.reshape(tf.slice(ices.indices, [0, 0], [-1, 1]), [-1])
                    ice_values = tf.string_to_number(ices.values, out_type=tf.int64)
                    ice_shape = tf.cast(tf.concat([tf.cast([ices.dense_shape[0]], tf.int32),
                                                   tf.constant([1], dtype=tf.int32),
                                                   tf.fill([1], self.dims[i + 1])], axis=0), tf.int64)
                    input_tensor = tf.SparseTensor(indices=tf.stack([ice_indices, tf.zeros_like(ice_indices,
                                                                                                dtype=tf.int64),
                                                                     ice_values], axis=-1),
                                                   values=tf.ones_like(ice_values, tf.float32),
                                                   dense_shape=ice_shape)

                if self.data_embedding[i] == 1:
                    with tf.variable_scope("embedding" + self.data_types[i + 1], reuse=embedding_reuse):
                        if self.data_types[i + 1] == 'ice':
                            ice_reuse = True
                        elif self.data_types[i + 1] == 'string':
                            str_reuse = True
                        elif self.data_types[i + 1] == 'web':
                            web_reuse = True

                        input_tensor = tf.sparse_reshape(input_tensor,
                                                         [input_tensor.dense_shape[0], input_tensor.dense_shape[2]])

                        w = tf.get_variable(name='w_' + 'embedding' + self.data_types[i + 1],
                                            shape=[self.dims[i + 1], self.embedding_dim],
                                            initializer=self.initializer(self.dims[i + 1], self.embedding_dim))
                        b = tf.get_variable(name='b_' + 'embedding' + self.data_types[i + 1],
                                            shape=[self.embedding_dim],
                                            initializer=tf.constant_initializer(0))

                        input_tensor = tf.sparse_tensor_dense_matmul(input_tensor, w)
                        input_tensor = tf.add(input_tensor, b)

                        if self.summaries_histogram:
                            tf.summary.histogram(name='w_' + 'embedding' + self.data_types[i + 1], values=w)
                            tf.summary.histogram(name='b_' + 'embedding' + self.data_types[i + 1], values=b)
                            with tf.variable_scope('norm_embedding' + self.data_types[i + 1], reuse=True):
                                tf.summary.histogram(name='gamma_n_embedding', values=tf.get_variable(name='gamma'))
                                tf.summary.histogram(name='beta_n_embedding', values=tf.get_variable(name='beta'))
                    all_dim += self.embedding_dim
                elif self.data_embedding[i] == 2:
                    with tf.variable_scope("att_emb" + self.data_types[i + 1], reuse=embedding_reuse):
                        if self.data_types[i + 1] == 'ice':
                            att_ice_reuse = True
                        elif self.data_types[i + 1] == 'string':
                            att_str_reuse = True
                        elif self.data_types[i + 1] == 'web':
                            att_web_reuse = True

                        ori_shape = input_tensor.dense_shape
                        input_tensor = tf.sparse_reshape(input_tensor,
                                                         [input_tensor.dense_shape[0] * input_tensor.dense_shape[1],
                                                          input_tensor.dense_shape[2]])

                        w = tf.get_variable(name='w_' + 'embedding' + self.data_types[i + 1],
                                            shape=[self.dims[i + 1], self.att_dim],
                                            initializer=self.initializer(self.dims[i + 1], self.att_dim))
                        b = tf.get_variable(name='b_' + 'embedding' + self.data_types[i + 1],
                                            shape=[self.att_dim],
                                            initializer=tf.constant_initializer(0))

                        input_tensor = tf.sparse_tensor_dense_matmul(input_tensor, w)
                        input_tensor = tf.add(input_tensor, b)
                        input_tensor = tf.reshape(input_tensor,
                                                  tf.stack([ori_shape[0], ori_shape[1],
                                                            tf.constant(self.att_dim, dtype=tf.int64)]))
                else:
                    all_dim += self.dims[i + 1]
                net.append(input_tensor)

        if self.data_embedding[0] == 2:
            with tf.variable_scope("attention", reuse=False):
                att1 = net[0]
                att2 = []
                for i in range(1, len(self.data_embedding)):
                    if self.data_embedding[i] == 2:
                        att2.append(net[i])
                att2 = tf.concat(att2, axis=1)
                att1, att2 = soft_attention_alignment(att1, att2)
                att1 = tf.layers.conv1d(att1, self.embedding_dim, 3, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer(self.att_dim, self.embedding_dim))
                att1 = tf.reduce_max(att1, axis=1)
                att2 = tf.layers.conv1d(att2, self.embedding_dim, 3, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer(self.att_dim, self.embedding_dim))
                att2 = tf.reduce_max(att2, axis=1)
            new_net = [att1, att2]
            all_dim = self.embedding_dim * 2
            for i in range(len(net)):
                if self.data_embedding[i] != 2:
                    new_net.append(net[i])
                if self.data_embedding[i] == 1:
                    all_dim += self.embedding_dim
                if self.data_embedding[i] == 0:
                    all_dim += self.dims[i + 1]
            net = new_net

        net = tf.concat(net, axis=-1)

        with tf.name_scope("cross"):
            cross0 = tf.expand_dims(net, -1)
            cross = tf.expand_dims(net, -1)
            for i in range(self.cross_layers):
                cross_w = tf.get_variable(name='cross_w' + str(i),
                                          shape=[all_dim, 1],
                                          initializer=self.initializer(all_dim, 1))
                cross_w = tf.tile(tf.expand_dims(cross_w, 0), [tf.shape(net)[0], 1, 1])
                cross_b = tf.get_variable(name='cross_b' + str(i),
                                          shape=[all_dim, 1],
                                          initializer=tf.constant_initializer(0))
                cross_l = cross
                cross = tf.matmul(cross0, tf.transpose(cross, perm=[0, 2, 1]))
                if self.double_cross:
                    cross_w2 = tf.get_variable(name='cross_w2' + str(i),
                                               shape=[all_dim, all_dim],
                                               initializer=self.initializer(all_dim, all_dim))
                    cross_w2 = tf.tile(tf.expand_dims(cross_w2, 0), [tf.shape(net)[0], 1, 1])
                    cross = tf.matmul(cross, cross_w2)
                    cross = tf.nn.relu(cross)
                cross = tf.add(tf.add(tf.matmul(cross, cross_w), cross_b), cross_l)
            cross = tf.reshape(cross, [-1, all_dim])

        with tf.name_scope("deep"):
            deep = net
            pre_dim = all_dim
            for i in range(self.deep_layers):
                deep = tf.contrib.layers.fully_connected(deep, self.deep_dims[i], activation_fn=tf.nn.relu,
                                                         weights_initializer=self.initializer(pre_dim,
                                                                                              self.deep_dims[i]),
                                                         scope="deep" + str(i))
                pre_dim = self.deep_dims[i]

        if self.deep_layers > 0 and self.cross_layers > 0:
            net = tf.concat([cross, deep], axis=1)
            all_dim += pre_dim
        elif self.deep_layers > 0:
            net = deep
            all_dim = pre_dim
        elif self.cross_layers > 0:
            net = cross

        with tf.name_scope("resnet"):
            resnets = []
            for i in range(self.num_res_layers):
                resnets.append(ResNetLayer(self.res_dims[i], all_dim, tf.nn.relu, self.initializer,
                                           True, self.summaries_histogram, is_training))
                net = resnets[-1](net, "resnet" + str(i))
            net = tf.contrib.layers.fully_connected(net, all_dim, activation_fn=None,
                                                    weights_initializer=self.initializer(all_dim, all_dim),
                                                    scope="tanh")
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training,
                                                name='norm_tanh', fused=None)
            net = tf.tanh(net)

        with tf.name_scope("sigmoid_layer"):
            net = tf.contrib.layers.fully_connected(net, 1, activation_fn=None,
                                                    weights_initializer=self.initializer(all_dim, 1),
                                                    scope="sigmoid_layer")
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training,
                                                name='norm_sigmoid', fused=None)
            score = tf.sigmoid(net)
            if self.summaries_histogram:
                with tf.variable_scope("sigmoid_layer", reuse=True):
                    tf.summary.histogram(name='w_sigmoid_layer', values=tf.get_variable(name='weights'))
                    tf.summary.histogram(name='b_sigmoid_layer', values=tf.get_variable(name='biases'))

        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=tf.expand_dims(input_tensors[0], axis=-1),
                                                     logits=net, pos_weight=self.pos_weight))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        tf.summary.scalar('loss', loss)

        label_true = tf.where(tf.greater(input_tensors[0], tf.ones_like(input_tensors[0]) * 0.5),
                              tf.ones_like(input_tensors[0]), tf.zeros_like(input_tensors[0]))
        score_true = tf.reshape(tf.where(tf.greater(score, tf.ones_like(score) * 0.5),
                                         tf.ones_like(score), tf.zeros_like(score)), [-1])
        TP = tf.where(tf.greater(tf.add(label_true, score_true), tf.ones_like(score_true) * 1.5),
                      tf.ones_like(score), tf.zeros_like(score))

        recall = tf.divide(tf.reduce_sum(TP), tf.reduce_sum(label_true) + 1e-10)
        tf.summary.scalar('recall', recall)
        precision = tf.divide(tf.reduce_sum(TP), tf.reduce_sum(score_true) + 1e-10)
        tf.summary.scalar('precision', precision)
        gradients = tf.norm(tf.gradients(loss, w))
        tf.summary.scalar('gradients', gradients)

        merged = tf.summary.merge_all()

        train_ops = {'loss': (loss, True), 'score': score, 'debug': debug, 'recall': recall,
                     'summary': merged, 'precision': precision, 'label': input_tensors[0]}
        train_ops.update(self.extend_op)

        predict_ops = {'loss': loss, 'score': score, 'recall': recall, 'summary': merged,
                       'precision': precision, 'label': input_tensors[0]}
        predict_ops.update(self.extend_op)

        placeholder = {m: d for m, d in zip(self.input_names, input)}
        placeholder['is_training'] = is_training_input

        return dict(train=mstf.Model.Action(train_ops, placeholder),
                    predict=mstf.Model.Action(predict_ops, placeholder))
