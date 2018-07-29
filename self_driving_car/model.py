import tensorflow as tf


def pilot_net(images, bins, mode):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = images

    net = tf.layers.conv2d(net, 24, [5, 5], strides = 2)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 36, [5, 5], strides = 2)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 48, [5, 5], strides = 2)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 64, [3, 3])
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 64, [3, 3])
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 200)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.dense(net, 100)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.dense(net, bins)

    return net


