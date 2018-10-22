import dataget as dg
import tensorflow as tf
import pandas as pd
import numpy as np
from model import pilot_net, cris_net, pilot_relation_net
from tensorflow.contrib import autograph
import tensorflow.contrib.slim as slim

def input_fn(data_dir, params):

    dataset = dg.data(
        "udacity-selfdriving-simulator",
        path = data_dir,
    )
    dataset = dataset.get()

    df = dataset.df

    df = process_dataframe(df, params)

    if params.only_center_camera:
        df = df[df.camera == 1]

    df = dg.shuffle(df)

    tensors = dict(
        filepath = df.filepath.as_matrix(),
        steering = df.steering.as_matrix(),
        camera = df.camera.as_matrix(),
        original_steering = df.original_steering.as_matrix(),
    )

    if "flipped" in df:
        tensors["flipped"] = df.flipped.as_matrix().astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(tensors)

    ds = ds.apply(tf.contrib.data.shuffle_and_repeat(
        buffer_size = params.buffer_size,
    ))

    ds = ds.apply(tf.contrib.data.map_and_batch(
        lambda row: process_data(row, params),
        batch_size = params.batch_size,
        num_parallel_calls = params.n_threads,
        drop_remainder = True,
    ))

    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
    
    return ds


def serving_input_fn(params):

    input_image = tf.placeholder(
        dtype = tf.float32,
        shape = [None, None, None, 3],
        name = "input_image",
    )

    images = tf.image.resize_images(input_image, [params.image_height, params.image_width])


    crop_window = get_crop_window(params)
    images = tf.image.crop_to_bounding_box(images, *crop_window)

    images = tf.image.resize_images(images, [params.resize_height, params.resize_width])

    images = (images / 255.0) * 2.0 - 1.0

    return tf.estimator.export.ServingInputReceiver(
        features = dict(
            image = images,
        ),
        receiver_tensors = dict(
            image = input_image,
        ),
    )



def model_fn(features, labels, mode, params):

    images = features["image"]

    if params.network == "pilot":
        network = pilot_net
    elif params.network == "cris":
        network = cris_net
    elif params.network == "relation":
        network = pilot_relation_net
    else:
        raise ValueError(params.network)

    predictions = network(
        images,
        mode,
        params, 
        conv_args = dict(
            kernel_regularizer = tf.contrib.layers.l2_regularizer(
                params.l2_regularization
            ),
        ),
    )

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                "serving_default" : tf.estimator.export.PredictOutput(predictions)
            }
        )

    onehot_labels = get_onehot_labels(features["steering"], params)

    

    tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels,
        logits = predictions["logits"],
        label_smoothing = params.label_smoothing,
        weights = get_weights(features["original_steering"], params)
    )

    loss = tf.losses.get_total_loss()

    labels = tf.argmax(onehot_labels, axis = 1)
    labels_pred = tf.argmax(predictions["logits"], axis = 1)

    if mode == tf.estimator.ModeKeys.EVAL:

        accuracy = tf.metrics.accuracy(labels, labels_pred)
        
        top_5_accuracy = tf.nn.in_top_k(
            predictions["logits"],
            labels,
            5,
        )
        top_5_accuracy = tf.cast(top_5_accuracy, tf.float32)
        top_5_accuracy = tf.metrics.mean(top_5_accuracy)

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss,
            eval_metric_ops = {
                "accuracy/top_5": top_5_accuracy,
                "accuracy/top_1": accuracy,
            }
        )

   

    if mode == tf.estimator.ModeKeys.TRAIN:

        with tf.name_scope("training"), tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            learning_rate = get_learning_rate(params)

            update = tf.contrib.opt.PowerSignOptimizer(
                learning_rate
            ).minimize(
                loss,
                global_step = tf.train.get_global_step()
            ) 

         # summaries
        accuracy = tf.contrib.metrics.accuracy(labels, labels_pred)

        top_5_accuracy = tf.nn.in_top_k(predictions["logits"], labels, 5)
        top_5_accuracy = tf.reduce_mean(tf.cast(top_5_accuracy, tf.float32))

        tf.summary.scalar("accuracy/top_1", accuracy)
        tf.summary.scalar("accuracy/top_5", top_5_accuracy)
        tf.summary.scalar("learning_rate", learning_rate)

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss,
            train_op = update,
        )


    
###############################
# helper functions
###############################

def get_weights(steering, params):


    ones = tf.ones_like(steering)
    zeros_weight = params.zeros_weight * ones

    weights = tf.where(
        tf.abs(steering) < 0.5,
        ones,
        ones * params.max_weight,
    )

    return tf.where(
        tf.equal(steering, 0.0),
        zeros_weight,
        weights,
    )

def get_crop_window(params):

    final_height = params.image_height - (params.crop_up + params.crop_down)
    final_width = params.image_width

    return [
        params.crop_up,
        0,
        final_height,
        final_width,
    ]

@autograph.convert()
def get_learning_rate(params):
    
    global_step = tf.train.get_global_step()

    initial_learning_rate = params.learning_rate * params.batch_size / 128.0

    if global_step < params.cold_steps:
        learning_rate = params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)
    
    elif global_step < params.cold_steps + params.warmup_steps:
        step = global_step - params.cold_steps
        p = step / params.warmup_steps
        
        learning_rate = initial_learning_rate * p + (1.0 - p) * params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)

    else:
        step = global_step - (params.cold_steps + params.warmup_steps)
        learning_rate = tf.train.linear_cosine_decay(initial_learning_rate, step, params.decay_steps, beta = params.final_learning_rate)
        learning_rate = tf.cast(learning_rate, tf.float32)

    return learning_rate


    

def get_onehot_labels(steering, params):
    
    label = tf.clip_by_value(steering, -1, 1)
    label = (label + 1.0) / 2.0
    label = label * (params.nbins - 1)

    label_upper = tf.ceil(label)
    label_lower = tf.floor(label)

    prob_upper = 1.0 - (label_upper - label)
    prob_upper = tf.cast(prob_upper, tf.float32)
    prob_upper = tf.expand_dims(prob_upper, 1)

    prob_lower = 1.0 - prob_upper

    onehot_upper = prob_upper * tf.one_hot(tf.cast(label_upper, tf.int32), params.nbins)
    onehot_lower = prob_lower * tf.one_hot(tf.cast(label_lower, tf.int32), params.nbins)

    onehot_labels = onehot_upper + onehot_lower

    onehot_labels = tf.Print(onehot_labels, 
        [
            onehot_labels[0],
            steering[0],
            label_upper[0],
            label_lower[0],
            prob_upper[0],
            prob_lower[0],
        ],
        first_n = 5,
    )

    return onehot_labels



def process_dataframe(df, params):

    df = df.copy()
    df_flipped = df.copy()

    df["flipped"] = False
    df_flipped["flipped"] = True

    df = pd.concat([df, df_flipped])

    df["original_steering"] = df.steering

    cam0 = df.camera == 0
    cam2 = df.camera == 2

    df.loc[cam0, "steering"] = df[cam0].steering + params.angle_correction
    df.loc[cam2, "steering"] = df[cam2].steering - params.angle_correction

    # flip
    flipped = df.flipped
    df.loc[flipped, "steering"] = -df[flipped].steering

    return df

def process_data(row, params):

    # read image
    image = tf.read_file(row["filepath"])
    image = tf.image.decode_and_crop_jpeg(
        contents = image,
        crop_window = get_crop_window(params),
        channels = 3,
    )

    image = tf.image.resize_images(image, [params.resize_height, params.resize_width])

    # print(row["flipped"])

    image = tf.cond(
        row["flipped"] > 0,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )

    if params.angle_noise_std > 0:
        noise = tf.random_normal([], mean = 0.0, stddev = params.angle_noise_std)
        row["steering"] = tf.cast(row["steering"], tf.float32) + noise
    else:
        row["steering"] = tf.cast(row["steering"], tf.float32)


    image = (image / 255.0) * 2.0 - 1.0

    row["image"] = image

    return row


def main():
    ds = input_fn("data/raw", {})
    print(ds)

if __name__ == '__main__':
    main()
