import data_pipeline
import model
import tensorflow as tf
import losses
import sys


ds = sys.argv[1]
effect = sys.argv[2]
outname = sys.argv[3]

train, train_size, test, test_size, val, val_size = data_pipeline.load_data_pipeline(ds, effect)
m = model.build_tcn_model()


m.compile(tf.keras.optimizers.Adam(
    learning_rate=0.00005
), loss=losses.multiwindow_stft_loss)

for i in range(3):
    m.fit(train, epochs=5, steps_per_epoch=train_size, validation_data=val, validation_steps = val_size)
    m.evaluate(test, steps=test_size)

    m.save("trained_models/"+effect+"/"+outname+"_epoch_"+str((i+1)*5))
