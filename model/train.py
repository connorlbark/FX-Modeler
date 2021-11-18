import data_pipeline
import model
import tensorflow as tf

train, train_size, test, test_size, val, val_size = data_pipeline.load_data_pipeline("data/copied_chorus_dataset", "Chorus3312")
m = model.build_tcn_model()

def fft_mse_loss(y_true, y_pred):
    # spectral representations for each signal
    y_true_fft = tf.signal.rfft(y_true)
    y_pred_fft = tf.signal.rfft(y_pred)

    # differences in magnitude/phase for each frequency
    # (abs converts complex spectral error to a real floating point by finding magnitude)
    spectral_errors = tf.math.abs(y_true_fft - y_pred_fft)

    # compute mean squared error
    squared_difference = tf.square(spectral_errors)
    return tf.reduce_mean(squared_difference) 


def fft_log_loss(y_true, y_pred):
    # spectral representations for each signal
    y_true_fft = tf.signal.rfft(y_true)
    y_pred_fft = tf.signal.rfft(y_pred)
    e = .001

    # get log difference of the _magnitude_ of each frequency.
    log_spectral_errors = tf.math.abs(tf.math.log(tf.math.abs(y_true_fft) + e) - tf.math.log(tf.math.abs(y_pred_fft) + e))

    # compute mean log error across all frequencies.
    return tf.reduce_mean(log_spectral_errors) 


m.compile(tf.keras.optimizers.SGD(
    learning_rate=0.01
), loss=fft_log_loss)

m.fit(train, epochs=5, steps_per_epoch=train_size, validation_data=val, validation_steps = val_size)
m.evaluate(test, steps=test_size)

m.save("trained_models/Chorus3312/model_11_17v3")
