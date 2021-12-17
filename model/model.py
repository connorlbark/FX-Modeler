from keras.layers.merge import add
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def add_tcn_block(input_layer, n_filters=16, kernel_size=4, dilations=[1,2,4,8]):

    tcn = input_layer
    #iteratively dilate upward
    for d in dilations:
        tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=d)(tcn)
        tcn = layers.BatchNormalization()(tcn)
        tcn = layers.Activation('relu')(tcn)
        tcn = layers.Dropout(0.3)(tcn)

    # residual skip connection

    input_correct_size = layers.Conv1D(n_filters, 1, padding='causal', activation='relu')(input_layer)
    tcn = layers.Add()([input_correct_size, tcn])

    return tcn


def add_tcn_block_reg(input_layer, n_filters=16, kernel_size=4, dilations=[1,2,4,8]):

    tcn = input_layer
    #iteratively dilate upward
    for d in dilations:
        tcn = layers.Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=d, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(tcn)
        tcn = layers.BatchNormalization()(tcn)
        tcn = layers.Activation('relu')(tcn)
        tcn = layers.Dropout(0.3)(tcn)

    # residual skip connection

    input_correct_size = layers.Conv1D(n_filters, 1, padding='causal', activation='relu')(input_layer)
    tcn = layers.Add()([input_correct_size, tcn])

    return tcn


def build_tcn_model_draft_v1():
    inputs = tf.keras.Input(shape=(None,2))
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=4, dilations=[1,2,4,8])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=6, dilations=[1,3,9,27])
    

    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=5, dilations=[1,10,100])
    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)





def build_tcn_model_reverb_v1(): # .001
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[10,100])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_rev_v2_v3(): # .001
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[2,4,8,16])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[4,16,64,256])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_rev_v4(): # .001
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[2,4,8,16,32])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100,1000])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[4,16,64,256])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_rev_v5(): # .001
    inputs = tf.keras.Input(shape=(None,2))
    
    #tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[1,2,4,8,16,32])
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[1,10,100,1000])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[1,4,16,64,256,512,1024])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_v6() :# .001
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block_reg(inputs, n_filters=32, kernel_size=13, dilations=[1,2,4,8,16,32])
    tcn = add_tcn_block_reg(inputs, n_filters=32, kernel_size=13, dilations=[1,10,100,1000])
    tcn = add_tcn_block_reg(tcn, n_filters=32, kernel_size=13, dilations=[1,4,16,64,256,512,1024])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)


def build_tcn_model_v7(): # .001
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[1,2,4,8,16,32])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[1,10,100,1000])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[1,4,16,64,256])

    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_v9(): # .0005
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[2,4,8,16,32])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100,1000])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[4,16,64,256])

    forward_in = layers.Conv1D(32, 1, padding='causal', activation='relu')(inputs)
    
    tcn = layers.Add()([tcn, forward_in])
    
    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model_rev_v10(): # 0.00005
    inputs = tf.keras.Input(shape=(None,2))    
    
    out = layers.Conv1D(2, 44100//2, padding='causal', activation='tanh')(inputs)
    return Model(inputs=inputs, outputs=out)

def build_tcn_model(): # 0.00005
    inputs = tf.keras.Input(shape=(None,2))
    
    tcn = add_tcn_block(inputs, n_filters=32, kernel_size=13, dilations=[2,4,8,16,32])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[10,100,1000])
    tcn = add_tcn_block(tcn, n_filters=32, kernel_size=13, dilations=[4,16,64,256])

    forward_in = layers.Conv1D(32, 1, padding='causal', activation='relu')(inputs)
    
    tcn = layers.Add()([tcn, forward_in])
    
    out = layers.Conv1D(2, 32, padding='causal', activation='tanh')(tcn)
    return Model(inputs=inputs, outputs=out)