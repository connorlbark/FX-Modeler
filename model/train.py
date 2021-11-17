import data_pipeline
import model
import tensorflow as tf

m = model.build_tcn_model()
ds, length = data_pipeline.load_data_pipeline("data/copied_chorus_dataset", "Chorus3312")

m.compile(tf.keras.optimizers.SGD(
    learning_rate=0.01
), 'mse')

m.fit(ds, epochs=10, steps_per_epoch=length)
