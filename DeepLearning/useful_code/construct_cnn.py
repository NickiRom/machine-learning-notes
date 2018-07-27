#################################################################
#
# the following code is for the construction of a standard CNN
# architecture for code testing purposes.  It is inspired by
# OKraus/DeepLoc git repository.  Code is under a BSD 3-Clause license.
# Copyright reproduced below:
# 
# Copyright (c) 2017, 
# All rights reserved.
#
# To construct a CNN:
# 
# import tensorflow as tf
# from construct_cnn import CNNModel
#
# input = tf.placeholder('float32', shape=[None,cropSize,cropSize,numChannels], name='input')
# is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
#
# logits = CNNModel(input, is_training)
# output = tn.nn.softmax(logits, name='softmax')
#
################################################################

import tensorflow as tf

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu,is_training=True,use_batch_norm=True):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram(layer_name + '/pre_activations', preactivate)
    if use_batch_norm:
      with tf.name_scope('batch_norm'):
        batch_norm = batch_norm_fc(preactivate,output_dim, phase_train=is_training,scope=layer_name+'_batch_norm')
        tf.summary.histogram(layer_name + '/batch_norm', batch_norm)

    else:
      batch_norm = preactivate
    if act:
        activations = act(batch_norm, name='activation')
    else:
        activations = batch_norm
    tf.summary.histogram(layer_name + '/activations', activations)
    return activations

def conv_layer(input_tensor, kernel_size_x, kernel_size_y,
               input_feat_maps, output_feat_maps, stride, layer_name, act=tf.nn.relu,is_training=True,use_batch_norm=True):
  """Reusable code for making a convolutional neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([kernel_size_x,kernel_size_y,input_feat_maps,output_feat_maps])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_feat_maps])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor,weights,
                                 strides=[1,stride,stride,1],padding='SAME') + biases
      tf.summary.histogram(layer_name + '/pre_activations', preactivate)
    if use_batch_norm:
      with tf.name_scope('batch_norm'):
        batch_norm = batch_norm_conv(preactivate, output_feat_maps, phase_train=is_training,scope=layer_name+'_batch_norm')
        tf.summary.histogram(layer_name + '/batch_norm', batch_norm)
    else:
      batch_norm = preactivate
    if act:
        activations = act(batch_norm, name='activation')
    else:
        activations = batch_norm
    tf.summary.histogram(layer_name + '/activations', activations)
    return activations

def pool2_layer(x,layer_name):
    with tf.name_scope(layer_name):
      pooled_activations = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding = 'SAME')
      tf.summary.histogram(layer_name + '/activations', pooled_activations)
    return pooled_activations
def pool3_layer(x,layer_name):
    with tf.name_scope(layer_name):
      pooled_activations = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1], padding = 'VALID')
      tf.summary.histogram(layer_name + '/activations', pooled_activations)
    return pooled_activations


def batch_norm_conv(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm_fc(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def CNNModel(input_images,is_training):

    conv1 = conv_layer(input_images, 3, 3, 2, 64, 1, 'conv_1',is_training=is_training)
    conv2 = conv_layer(conv1, 3, 3, 64, 64, 1, 'conv_2', is_training=is_training)
    pool1 = pool2_layer(conv2, 'pool1')
    conv3 = conv_layer(pool1, 3, 3, 64, 128, 1, 'conv_3', is_training=is_training)
    conv4 = conv_layer(conv3, 3, 3, 128, 128, 1, 'conv_4', is_training=is_training)
    pool2 = pool2_layer(conv4, 'pool2')
    conv5 = conv_layer(pool2, 3, 3, 128, 256, 1, 'conv_5', is_training=is_training)
    conv6 = conv_layer(conv5, 3, 3, 256, 256, 1, 'conv_6', is_training=is_training)
    conv7 = conv_layer(conv6, 3, 3, 256, 256, 1, 'conv_7', is_training=is_training)
    conv8 = conv_layer(conv7, 3, 3, 256, 256, 1, 'conv_8', is_training=is_training)
    pool3 = pool2_layer(conv8, 'pool3')
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 256])
    fc_1 = nn_layer(pool3_flat, 8 * 8 * 256, 512, 'fc_1', act=tf.nn.relu, is_training=is_training)
    fc_2 = nn_layer(fc_1, 512, 512, 'fc_2', act=tf.nn.relu,is_training=is_training)
    logit = nn_layer(fc_2, 512, 19, 'final_layer', act=None, is_training=is_training)

    return logit



