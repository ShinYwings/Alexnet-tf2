import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'places365_small',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir=r"D:"
)