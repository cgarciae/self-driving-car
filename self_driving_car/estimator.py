import dataget as dg
import tensorflow as tf

def input_fn(data_dir, params):

    dataset = dg.data(
        "udacity-selfdriving-simulator",
        path = data_dir,
    )
    dataset = dataset.get()

    df = dataset.df

    df = dg.shuffle(df)


    ds = tf.data.Dataset.from_tensor_slices(dict(
        filename = df.filename.as_matrix(),
        steering = df.steering.as_matrix(),
        camera = df.camera.as_matrix(),
    ))

    
    return ds


def main():
    ds = input_fn("data/raw", {})
    print(ds)

if __name__ == '__main__':
    main()
