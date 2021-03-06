from imgaug import augmenters as iaa
from dask import dataframe as dd
import dataget as dg
import numpy as np
from dask.diagnostics import ProgressBar
import pandas as pd
import click
import dicto as do
import os
from PIL import Image
from dask.multiprocessing import get
import shutil

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "config", "data-augmentation.yml")
SLASH = os.sep

@click.command()
@click.option("--raw-dir", required = True)
@click.option("--augmented-dir", required = True)
@click.option("--limit", default = None, type = int)
@click.option("--rm", is_flag = True)
@do.click_options_config(PARAMS_PATH, "params", underscore_to_dash = False)
def main(raw_dir, augmented_dir, limit, rm, params):

    if rm and os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    
    df = dg.data(
        "udacity-selfdriving-simulator",
        path = raw_dir,
        normalize = False,
    ).df

    if limit:
        df = df.sample(n=limit)

    sd = augment_dataset(df, params, save_dir=augmented_dir, return_image=False)

    print("AUGMENTING IMAGES")
    with ProgressBar():
        df = sd.compute(get=get)

    

    print("SAVING CSVs")
    for _, dfg in df.groupby(["folder", "augment_idx"]):

        sample = dfg.iloc[0]
        folder = sample.folder + "_" + str(sample.augment_idx)
        csv_path = os.path.join(augmented_dir, folder, "driving_log.csv")

        del dfg["left_image"]
        del dfg["left_filepath"]
        del dfg["center_image"]
        del dfg["center_filepath"]
        del dfg["right_image"]
        del dfg["right_filepath"]

        del dfg["augment_idx"]
        del dfg["folder"]
        del dfg["augmented"]

        dfg.to_csv(csv_path, index = False)
        


def get_seq(params):

    filters = iaa.SomeOf(params.filters_repeat, [
        iaa.ChangeColorspace("BGR"),
        iaa.ChangeColorspace("GRAY"),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 9)),
        iaa.MedianBlur(k=(3, 9)),
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.Add((-40, 40)),
        iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.Multiply((0.5, 1.5)),
        iaa.MultiplyElementwise((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25),
        iaa.Sharpen(alpha=(0.6, 1.0)),
        iaa.Emboss(alpha=(0.0, 0.5)),
        iaa.CoarseDropout(0.2, size_percent=0.00001, per_channel = 1.0),
    ])
    affine = iaa.Affine(
        rotate=(-7, 7),
        scale=(0.9, 1.1),
        translate_percent=dict(x = (-0.05, 0.05)),
        mode = "symmetric",
    )

    return iaa.Sequential([
        filters,
        affine,
    ])

def load_image(row, path_column, name_column, seq, save_dir = None, return_image = True):

    
    image = dg.functions._load_image(row[path_column])

    

    if row["augmented"]:
        try:
            image = seq.augment_image(image)
        except:
            print(row)
            raise

    
    if save_dir is not None:
        folder = row.folder + "_" + str(row.augment_idx)
        image_folder = os.path.join(save_dir, folder, "IMG")
        image_path = os.path.join(image_folder, row[name_column].strip())
        
        os.makedirs(image_folder, exist_ok=True)

        im = Image.fromarray(image)
        im.save(image_path)
    
    if return_image:
        return (image,)
    else:
        return None
    


def augment_dataset(df, params, save_dir = None, return_image = True):

    seq = get_seq(params)

    df = df.copy()

    df["augmented"] = False
    df["augment_idx"] = 0


    dfs = [df]
    for i in range(params.augmentation_factor - 1):
        dfi = df.copy()
        dfi["augmented"] = True
        dfi["augment_idx"] = i + 1
        dfs.append(dfi)

    df = pd.concat(dfs)

    sd = dd.from_pandas(df, npartitions = params.n_threads)

    if "filepath" in sd.columns:
        sd["image"] = sd.apply(lambda row: load_image(row, "filepath", "filename", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
    else:
        sd["left_image"] = sd.apply(lambda row: load_image(row, "left_filepath", "left", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
        sd["center_image"] = sd.apply(lambda row: load_image(row, "center_filepath", "center", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
        sd["right_image"] = sd.apply(lambda row: load_image(row, "right_filepath", "right", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)

    return sd

    


if __name__ == '__main__':
    main()



    

    

    


