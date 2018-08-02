import os
import tarfile

# set environment variables so that click works
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

import tensorflow as tf
import click
import logging
import dicto as do
from . import estimator as est
from datetime import datetime


PARAMS_PATH = os.path.join(os.path.dirname(__file__), "config", "params.yml")
PROJECT = "pilotnet"

class Mode:
    train_and_evaluate = "train_and_evaluate"
    train = "train"
    evaluate = "evaluate"
    export = "export"


@click.command()
@click.option('--data-dir', required = True)
@click.option('--job-dir', required = True)
@click.option('--mode', default = "train_and_evaluate", type=click.Choice([Mode.train_and_evaluate, Mode.train, Mode.evaluate, Mode.export]))
@do.click_options_config(PARAMS_PATH, "params", underscore_to_dash = False)
def main(data_dir, job_dir, mode, params):
    
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
            PROJECT,
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
        os.path.join(job_dir, "export", PROJECT),
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
    main()