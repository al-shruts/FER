import time
import numpy as np
import pandas as pd

from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, min_delta=1e-1, verbose=1, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir="./logs/", histogram_freq=1, update_freq='batch')

SHAPE = (224, 224)
NUM_CLASSES = 9
FT_UP_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 10
DF_FRAC = 0.1

df = pd.read_csv('data/train.csv', index_col=0)
df = df.sample(frac=DF_FRAC)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./data',
    x_col='image_path',
    y_col='emotion',
    target_size=SHAPE,
    batch_size=BATCH_SIZE,
    subset='training',
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./data',
    x_col='image_path',
    y_col='emotion',
    target_size=SHAPE,
    batch_size=BATCH_SIZE,
    subset='validation',
)

test_generator = test_datagen.flow_from_directory(
    './data/test',
    target_size=(224, 224),
    class_mode=None,
    shuffle=False,
    batch_size=1
)

base_model = VGGFace(model='vgg16', weights='vggface', include_top=False, input_shape=(*SHAPE, 3))
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(
    units=hp.Int('dense_1_units', min_value=128, max_value=1024, step=64),
    activation='relu'
)(x)
x = Dropout(
    rate=hp.Float('dropout_1_rate', min_value=0.0, max_value=1, step=0.1)
)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(
        hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    ),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.fit(train_generator, epochs=hp.Int('epochs', 5, 100, 5), validation_data=validation_generator, callbacks=[early_stop])


