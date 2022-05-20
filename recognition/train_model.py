import tensorflow as tf
from tensorflow import keras
from keras import layers

image_size = (180, 180)
batch_size = 20

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "recognition/dataset",
    validation_split=0.2,
    subset="training",
    seed=1332,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "recognition/dataset",
    validation_split=0.2,
    subset="validation",
    seed=1332,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(buffer_size=20)
val_ds = val_ds.prefetch(buffer_size=20)


def make_model(input_shape, num_classes):
    return keras.models.Sequential([
        layers.Rescaling(1 / 255, input_shape=input_shape),

        layers.Input(input_shape),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.GlobalMaxPool2D(),
        layers.Dense(num_classes, activation="sigmoid")
    ])


model = make_model(input_shape=image_size + (3,), num_classes=1)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
