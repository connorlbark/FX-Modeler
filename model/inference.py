import argparse
from posixpath import normpath
import soundfile
from tensorflow import keras
import tensorflow as tf
import losses

# --------------------------- ARGUMENTS ---------------------------
parser = argparse.ArgumentParser(description='Generate dataset.')

parser.add_argument('modelpath', help='path to model')

parser.add_argument('inpath', help='path to wav file.')
parser.add_argument('outpath', help='path to ouputted file with effect.')
args = parser.parse_args()


m = keras.models.load_model(args.modelpath, custom_objects={
    "stft_loss": losses.stft_loss,
    "multiwindow_stft_loss": losses.multiwindow_stft_loss,
})

in_audio = tf.io.read_file(args.inpath)
in_audio, sample_rate = tf.audio.decode_wav(in_audio,
                                  desired_channels=2)

in_audio_batched = tf.expand_dims(in_audio, 0)

out_audio_batched = m.call(in_audio_batched)
out_audio = tf.squeeze(out_audio_batched)


out_file = tf.audio.encode_wav(out_audio,
                                  sample_rate)

tf.io.write_file(args.outpath, out_file)