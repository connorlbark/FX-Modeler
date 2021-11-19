import tensorflow as tf

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

def stft_log_spectral_magnitude(y_true_mag, y_pred_mag):
    # get log difference of the _magnitude_ of each frequency.
    return tf.abs(tf.math.log(y_true_mag) - tf.math.log(y_pred_mag))


def stft_spectral_convergence(y_true_mag, y_pred_mag):
    # spectral representations for each signal
    return tf.norm(y_true_mag - y_pred_mag, ord='fro', axis=(-2,-1)) / tf.norm(y_true_mag, ord='fro', axis=(-2,-1))

def stft_loss(y_true, y_pred):
    frame_length=1024
    frame_step=600
    fft_length=1024
    y_true_stft = tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, pad_end=True)
    y_pred_stft = tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, pad_end=True)

    y_true_stft_abs = tf.abs(y_true_stft)
    y_pred_stft_abs = tf.abs(y_pred_stft)


    y_true_stft_abs = tf.clip_by_value(tf.math.sqrt(y_true_stft_abs ** 2 + 1e-7), 1e-7, 1e3)
    y_pred_stft_abs = tf.clip_by_value(tf.math.sqrt(y_pred_stft_abs ** 2 + 1e-7), 1e-7, 1e3)


    log_mag = stft_log_spectral_magnitude(y_true_stft_abs, y_pred_stft_abs)
    convergence = stft_spectral_convergence(y_true, y_pred)
    return tf.reduce_mean(log_mag) + tf.reduce_mean(convergence)