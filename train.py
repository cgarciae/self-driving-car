import os
import tarfile

# set environment variables so that click works
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

import tensorflow as tf
import fire
import logging
import dicto as do
import estimator as est
from datetime import datetime



class Mode:
    train_and_evaluate = "train_and_evaluate"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    

PARAMS = dict(
    network = "pilot",
    project = "pilotnet",

    # bins
    nbins = 51,

    # dataset
    angle_correction = 0.05,
    only_center_camera = False,

    # regularization
    label_smoothing = 0.0,
    zeros_weight = 0.3,
    max_weight = 5.0,
    angle_noise_std = 0.0,
    l1_embeddings_regularization = 0.0,
    l2_regularization = 0.0,
    dropout = 0.15,

    # train
    summary_steps = 200,
    max_steps = 200000,
    save_checkpoints_steps = 5000,
    eval_steps = 20,
    start_delay_secs = 60,
    throttle_secs = 120,

    # pipeline
    batch_size = 64,
    buffer_size = 1000,
    n_threads = 4,

    # image
    image_height = 160,
    image_width = 320,
    crop_up = 50,
    crop_down = 25,
    resize_height = 66,
    resize_width = 200,

    # learning rate
    cold_learning_rate =  0.000001,
    learning_rate = 0.01,
    final_learning_rate = 0.0001,

    cold_steps = 0,
    warmup_steps = 0,
)

PARAMS["decay_steps"] = PARAMS["max_steps"]

@do.fire_options(PARAMS, "params")
def main(data_dir, job_dir, mode, params):

    print("JOB_DIR:", job_dir)
    
    tf.logging.set_verbosity(tf.logging.INFO)

    run_config = tf.estimator.RunConfig(
        model_dir = job_dir,
        save_summary_steps = params.summary_steps,
        save_checkpoints_steps = params.save_checkpoints_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn = est.model_fn,
        params = params,
        config = run_config,
    )


    if mode == Mode.train_and_evaluate:

        # save everything
        compress_code(job_dir)

        exporter = tf.estimator.LatestExporter(
            params.project,
            lambda: est.serving_input_fn(params)
        )

        train_spec = tf.estimator.TrainSpec(
            lambda: est.input_fn(data_dir, params),
            max_steps = params.max_steps,
        )
        
        test_spec = tf.estimator.EvalSpec(
            lambda: est.input_fn(data_dir, params),
            steps = params.eval_steps,
            exporters = [exporter],
        )

        # main & evaluate
        tf.logging.info("Start train_and_evalutate...")
        tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)

    elif mode == Mode.train:
        # save everything
        compress_code(job_dir)

        # main & evaluate
        tf.logging.info("Start train...")
        estimator.train(
            input_fn = lambda: est.input_fn(data_dir, params),
            max_steps = params.max_steps,
        )

    elif mode == Mode.evaluate:
        tf.logging.info("Start evaluate...")
        estimator.evaluate(
            input_fn = lambda: est.input_fn(data_dir, params),
            steps = params.eval_steps,
        )
    
    elif mode == Mode.export:
        pass

    else:
        raise ValueError("Mode '{mode}' not supported".format(mode = mode))

    # export
    tf.logging.info("Exporting Saved Model: ")
    estimator.export_savedmodel(
        os.path.join(job_dir, "export", params.project),
        lambda: est.serving_input_fn(params)
    )

def compress_code(job_dir):
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    dirname = os.path.dirname(__file__)
    packages_dir = os.path.join(job_dir, "code")
    tar_path = os.path.join(packages_dir, date + ".tar.gz")

    os.makedirs(packages_dir, exist_ok=True)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dirname, arcname=dirname)


if __name__ == '__main__':
    fire.Fire(main)