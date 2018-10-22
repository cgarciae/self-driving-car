import tensorflow as tf
import tfinterface as ti
import numpy as np


def pilot_net(images, mode, params, conv_args = {}):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = images

    net = tf.layers.conv2d(net, 24, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 36, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 48, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 200, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.dense(net, 100, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.dense(net, params.nbins)

    return dict(
        logits = net,
        probabilities = tf.nn.softmax(net),
    )

def pilot_relation_net(images, mode, params, conv_args = {}):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = images

    net = tf.layers.conv2d(net, 24, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 36, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 48, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    # get relations
    net = ti.layers.add_coordinates(net)

    n_objetcs = np.prod(net.shape[1:-1])
    n_channels = net.shape[1]
    
    net = tf.reshape(net, [-1, n_channels])

    net = tf.layers.dense(net, 200, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    # aggregate relations
    n_channels = net.shape[1]
    net = tf.reshape(net, [-1, n_objetcs, n_channels])
    net = tf.reduce_max(net, axis = 1)

    # calculate global attribute
    net = tf.layers.dense(net, params.nbins)

    return dict(
        logits = net,
        probabilities = tf.nn.softmax(net),
    )


def cris_net(images, mode, params, conv_args = {}):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = images

    net = tf.layers.conv2d(net, 24, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 36, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 48, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)


    net = tf.squeeze(net, axis = 1)
    

    net = tf.layers.conv1d(net, 16, [3], padding="SAME", **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv1d(net, 4, [3], padding="SAME", **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv1d(net, 1, [3], padding="SAME", **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)


    net = embedding = tf.layers.flatten(net)

    # net = tf.layers.dense(net, 200)
    # net = tf.layers.batch_normalization(net, training=training)
    # net = tf.nn.relu(net)
    # net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.dense(net, 100, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.dense(net, params.nbins)

    # losses
    if params.l1_embeddings_regularization > 0:
        embedding_loss = tf.contrib.layers.l1_regularizer(scale = params.l1_embeddings_regularization)(embedding)
        tf.losses.add_loss(embedding_loss)

    return dict(
        logits = net,
        probabilities = tf.nn.softmax(net),
        embedding = embedding,
    )