from keras.layers.merge import add
import tensorflow as tf
from tensorflow.keras import layers, Model

def add_tcn_block(input_layer, n_filters=16, kernel_size=4, dilations=[1,2,4,8]):

    tcn = input_layer
    #iteratively dilate upward
    for d in dilations:
        tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=d)(tcn)
        tcn = layers.BatchNormalization()(tcn)
        tcn = layers.Activation('relu')(tcn)
        tcn = layers.Dropout(0.2)(tcn)

    # residual skip connection

    input_correct_size = layers.Conv1D(n_filters, 1, padding='causal', activation='relu')(input_layer)
    tcn = layers.Add()([input_correct_size, tcn])

    return tcn


def build_tcn_model():
    inputs = tf.keras.Input(shape=(None,2))
    #tcn = layers.Conv1D(16, 1, padding='causal', dilation_rate=1, activation='relu')(inputs)
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=4, dilations=[1,2,4,8])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=4, dilations=[1,10,100])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=4, dilations=[1,2,4,8])
    
    # tcn = add_tcn_block(tcn, 32, 4)
    out = layers.Conv1D(2, 4, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)
