from keras.layers.merge import add
import tensorflow as tf
from tensorflow.keras import layers, Model

def add_tcn_block(input_layer, n_filters=32, kernel_size=4):
    # convolve d=1
    tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=1, activation='relu')(input_layer)
    tcn = layers.BatchNormalization()(tcn)
    # convolve d=2
    tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=2, activation='relu')(tcn)
    tcn = layers.BatchNormalization()(tcn)

    # convolve d=4
    tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=4, activation='relu')(tcn)
    tcn = layers.BatchNormalization()(tcn)

    # convolce d=8
    tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=8, activation='relu')(tcn)
    tcn = layers.BatchNormalization()(tcn)

    # residual skip connection
    tcn = layers.Add()([input_layer, tcn])

    return tcn


def build_tcn_model():
    inputs = tf.keras.Input(shape=(None,2))
    tcn = layers.Conv1D(32, 4, padding='causal', dilation_rate=1, activation='relu')(inputs)
    tcn = add_tcn_block(tcn, 32, 4)
    # tcn = add_tcn_block(tcn, 32, 4)
    out = layers.Conv1D(2, 4, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)
