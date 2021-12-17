import tensorflow as tf
from os import listdir
from os.path import join

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _load_audio(nofx_path, fx_path):
    # Load one second of audio at 44.1kHz sample-rate
    nofx_audio = tf.io.read_file(nofx_path)
    nofx_audio, _ = tf.audio.decode_wav(nofx_audio,
                                             desired_channels=2)
 
    fx_audio = tf.io.read_file(fx_path)
    fx_audio, _ = tf.audio.decode_wav(fx_audio,
                                                desired_channels=2)
    return nofx_audio, fx_audio


def prepadded_batch(ds, batch_size, padding_values=None):
    windowed = ds.window(batch_size)
    if padding_values == None:
        padding_values = 0.0
    

    
    # def _pad(ds1, ds2):
    #     vals = []
        
    #     for val in ds1:
    #         vals.append(val)
    #    # eager_values = list(ds.as_numpy_iterator())

    #     return tf.keras.preprocessing.sequence.pad_sequences(
    #             vals, padding='pre',
    #             truncating='pre', value=padding_values
    #         )

    def _max_size_reduce_fn(state, val):
        shape = tf.shape(val)
        size = shape[-1]
        
        if state > size:
            return state
        return size
    
    largest_tensor = windowed.reduce(0, _max_size_reduce_fn )
    
    print(largest_tensor)
    # uniform_size = windowed.map()
    
    # def _stack_reduce_fn(state, val):
        
    
    # return uniform_size.reduce(None, _stack_reduce_fn)

def _load_dataset_as_audio(ds, shuffle_buffer_size=1024, batch_size=1):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=25)
    # Load and decode audio from file paths
    ds = ds.map(_load_audio, num_parallel_calls=AUTOTUNE)
    # Repeat dataset forever
    ds = ds.repeat()
    # batch
    ds = ds.batch(batch_size)
    # ds = prepadded_batch(ds, 32)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def _load_cleaned_dataset_label_mapping(dataset_path, fx_class):

    return     [
         join(dataset_path, "NoFX1111", datafile)
        for 
            datafile
        in
            listdir(join(dataset_path, fx_class)) 
    ],     [
        join(dataset_path, fx_class, datafile)
        for 
            datafile
        in
            listdir(join(dataset_path, fx_class)) 
    ]

def _split_into_train_test(ds, ds_size, train_split=0.8, test_split=0.1, val_split=0.1):
    ds = ds.shuffle(ds_size, seed=12)
    
    train_size = int(train_split * ds_size)
    test_size = int(test_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)    
    test_ds = ds.skip(train_size).take(test_size)
    val_ds = ds.skip(train_size).skip(test_size)

    return train_ds, train_size, test_ds, test_size, val_ds, val_size



def load_data_pipeline(dataset_path, fx_class):
    nofxpaths, fxclasspaths = _load_cleaned_dataset_label_mapping(dataset_path, fx_class)

    ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(nofxpaths), 
                              tf.data.Dataset.from_tensor_slices(fxclasspaths)))
    
    train, train_size, test, test_size, val, val_size = _split_into_train_test(ds, len(nofxpaths))

    return _load_dataset_as_audio(train), train_size, _load_dataset_as_audio(test), test_size, _load_dataset_as_audio(val), val_size
