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

        del dfg["image"]
        del dfg["augment_idx"]
        del dfg["filepath"]
        del dfg["folder"]

        dfg.to_csv(csv_path, index = False)
        


def get_seq():
    return iaa.Sequential([
        iaa.OneOf([
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
        ]),
        iaa.Affine(
            rotate=(-7, 7),
            scale=(0.9, 1.1),
            translate_percent=dict(x = (-0.05, 0.05)),
            mode = "symmetric",
        ),
    ])

def load_image(row, seq, save_dir = None, return_image = True):

    
    image = dg.functions._load_image(row.filepath)

    

    if row["augment"]:
        try:
            image = seq.augment_image(image)
        except:
            print(row)
            raise

    
    if save_dir is not None:
        folder = row.folder + "_" + str(row.augment_idx)
        image_folder = os.path.join(save_dir, folder, "IMG")
        image_path = os.path.join(image_folder, row.filename.strip())
        
        os.makedirs(image_folder, exist_ok=True)

        im = Image.fromarray(image)
        im.save(image_path)
    
    if return_image:
        return (image,)
    else:
        return None
    


def augment_dataset(df, params, save_dir = None, return_image = True):

    seq = get_seq()

    df = df.copy()

    df["augment"] = False
    df["augment_idx"] = 0


    dfs = [df]
    for i in range(params.augmentation_factor - 1):
        dfi = df.copy()
        dfi["augment"] = True
        dfi["augment_idx"] = i + 1
        dfs.append(dfi)

    df = pd.concat(dfs)

    sd = dd.from_pandas(df, npartitions = params.n_threads)

    sd["image"] = sd.apply(lambda row: load_image(row, seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)

    return sd

    


if __name__ == '__main__':
    main()



    

    

    

