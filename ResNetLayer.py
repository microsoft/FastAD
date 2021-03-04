#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/6/2018 3:31 PM
# @Author  : Jianjin Zhang
# @File    : ResNetLayer.py

import tensorflow as tf


class ResNetLayer(object):
    def __init__(self, inner_dim, out_dim, activation, initializer, batch_norm, summaries_histogram, is_training):
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.activation = activation
        self.initializer = initializer
        self.batch_norm = batch_norm
        self.summaries_histogram = summaries_histogram
        self.is_training = is_training

    def __call__(self, x, name):
        with tf.name_scope(name):
            net = tf.contrib.layers.fully_connected(x, self.inner_dim, activation_fn=None,
                                                    weights_initializer=self.initializer(self.out_dim, self.inner_dim),
                                                    scope="l1" + name)
            net = tf.layers.batch_normalization(net, axis=-1, training=self.is_training, name="n1" + name, fused=None)
            net = self.activation(net)

            net = tf.contrib.layers.fully_connected(net, self.out_dim, activation_fn=None,
                                                    weights_initializer=self.initializer(self.inner_dim, self.out_dim),
                                                    scope="l2" + name)
            net = tf.layers.batch_normalization(net, axis=-1, training=self.is_training, name="n2" + name, fused=None)
            output = tf.nn.relu(tf.add(net, x))

            if self.summaries_histogram:
                with tf.variable_scope("l1" + name, reuse=True):
                    tf.summary.histogram(name='w_' + 'l1' + name, values=tf.get_variable(name='weights'))
                    tf.summary.histogram(name='b_' + 'l1' + name, values=tf.get_variable(name='biases'))
                with tf.variable_scope("n1" + name, reuse=True):
                    tf.summary.histogram(name='gamma_' + 'n1' + name, values=tf.get_variable(name='gamma'))
                    tf.summary.histogram(name='beta_' + 'n1' + name, values=tf.get_variable(name='beta'))
                with tf.variable_scope("l2" + name, reuse=True):
                    tf.summary.histogram(name='w_' + 'l2' + name, values=tf.get_variable(name='weights'))
                    tf.summary.histogram(name='b_' + 'l2' + name, values=tf.get_variable(name='biases'))
                with tf.variable_scope("n2" + name, reuse=True):
                    tf.summary.histogram(name='gamma_' + 'n2' + name, values=tf.get_variable(name='gamma'))
                    tf.summary.histogram(name='beta_' + 'n2' + name, values=tf.get_variable(name='beta'))

        return output
