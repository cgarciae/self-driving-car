import tensorflow as tf


def pilot_net(images, mode, params):

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

    net = tf.layers.dense(net, params.nbins)

    return dict(
        logits = net,
        probabilities = tf.nn.softmax(net),
    )


def cris_net(images, mode, params):

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


    net = tf.squeeze(net, axis = 1)
    

    net = tf.layers.conv1d(net, 16, [3], padding="SAME")
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv1d(net, 4, [3], padding="SAME")
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv1d(net, 1, [3], padding="SAME")
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)


    net = embedding = tf.layers.flatten(net)

    # net = tf.layers.dense(net, 200)
    # net = tf.layers.batch_normalization(net, training=training)
    # net = tf.nn.relu(net)

    net = tf.layers.dense(net, 100)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

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