import data_pipeline
import model
import tensorflow as tf

train, train_size, test, test_size, val, val_size = data_pipeline.load_data_pipeline("data/copied_chorus_dataset", "Chorus3312")
m = model.build_tcn_model()

m.compile(tf.keras.optimizers.SGD(
    learning_rate=0.01
), 'mse')

m.fit(train, epochs=1, steps_per_epoch=train_size, validation_data=val, validation_steps = val_size)

m.evaluate(test, steps=test_size)

m.save("trained_models/Chorus3312/model_11_17v1")
