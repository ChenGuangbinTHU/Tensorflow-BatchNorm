# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = is_train

        x = tf.reshape(self.x_, [-1, 28, 28, 1])
        W_conv1 = weight_variable([3, 3, 1, 4])
        b_conv1 = bias_variable([4])
        h_conv1 = tf.nn.relu(batch_normalization_layer(conv2d(x, W_conv1) + b_conv1, 4, self.is_train))
        h_pool1 = pooling(h_conv1)

        print(h_pool1.get_shape())

        W_conv2 = weight_variable([4, 4, 4, 4])
        b_conv2 = bias_variable([4])
        h_conv2 = tf.nn.relu(batch_normalization_layer(conv2d(h_pool1, W_conv2) + b_conv2, 4, self.is_train))
        h_pool2 = pooling(h_conv2)

        print(h_pool2.get_shape())

        h_pool2_reshape = tf.reshape(h_pool2, [-1,4*5*5])
        W_l = weight_variable([4*5*5, 10])
        b_l = bias_variable([10])
        logits = tf.matmul(h_pool2_reshape, W_l) + b_l

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, [1,1,1,1], padding='VALID')

def pooling(x) :
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, out_size, isTrain=True):
    # in_size, out_size = inputs.get_shape()
    pop_mean = tf.Variable(tf.zeros([out_size]),trainable=False)
    pop_var = tf.Variable(tf.ones([out_size]),trainable=False)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    eps = 0.001
    decay = 0.999
    if isTrain:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        print(batch_mean.get_shape())
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean,batch_var,shift,scale,eps)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,pop_var,shift,scale,eps)

# def batch_normalization_layer(inputs, isTrain=True):
#     # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
#     # hint: you can add extra parameters (e.g., shape) if necessary
#     return inputs
