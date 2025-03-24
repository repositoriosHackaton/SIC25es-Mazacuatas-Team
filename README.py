# SIC25es-Mazacuatas-Team
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

script_dir = os.getcwd()
train_dir = os.path.join(script_dir, 'TrashDataset', 'train')
test_dir = os.path.join(script_dir, 'TrashDataset', 'test')

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Las rutas del dataset no son válidas. Verifica las rutas de 'train' y 'test'.")

modelo_base = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in modelo_base.layers:
    layer.trainable = False

for layer in modelo_base.layers[-30:]:
    layer.trainable = True

modelo_nuevo = models.Sequential([
    modelo_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])

modelo_nuevo.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.3,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    seed=123
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    seed=123
)

class_weights = {0: 1.0, 1: 1.0, 2: 2.0, 3: 2.0, 4: 1.5, 5: 1.5}

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = modelo_nuevo.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

modelo_nuevo_path = os.path.join(script_dir, 'modelo_clasificacion_basura_v3.h5')
modelo_nuevo.save(modelo_nuevo_path)

print(f"Modelo guardado en: {modelo_nuevo_path}")
