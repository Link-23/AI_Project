# Importar librerias
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rutas de los directorios de datos
entrenamiento = r"Ingresar la ruta de la capetas con los archvos para entrenar la ia"
validacion = r"Ingresar la ruta de la carpeta con los archivos para la Validacion de la ia"

# Parámetros
ancho, alto = 200, 200

# Lista de etiquetas
etiquetas = []

# Listas para almacenar imágenes y etiquetas
fotos_entrenamiento = []
etiquetas_entrenamiento = []

# Lectura de datos de entrenamiento
for nombreDir in os.listdir(entrenamiento):
    etiquetas.append(nombreDir)
    ruta = os.path.join(entrenamiento, nombreDir)
    for fileName in os.listdir(ruta):
        img = cv2.imread(os.path.join(ruta, fileName), 0)
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(ancho, alto, 1)
        fotos_entrenamiento.append(img)
        etiquetas_entrenamiento.append(etiquetas.index(nombreDir))

# Normalizar las imágenes de entrenamiento
fotos_entrenamiento = np.array(fotos_entrenamiento).astype(float) / 255

# Convertir etiquetas de entrenamiento a formato numérico
etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)

# Lectura de datos de validación
fotos_validacion = []
etiquetas_validacion = []

for nombreDir in os.listdir(validacion):
    ruta = os.path.join(validacion, nombreDir)
    for fileName in os.listdir(ruta):
        img = cv2.imread(os.path.join(ruta, fileName), 0)
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(ancho, alto, 1)
        fotos_validacion.append(img)
        etiquetas_validacion.append(etiquetas.index(nombreDir))

# Normalizar las imágenes de validación
fotos_validacion = np.array(fotos_validacion).astype(float) / 255

# Convertir etiquetas de validación a formato numérico
etiquetas_validacion = np.array(etiquetas_validacion)

# Crear generador de imágenes aumentadas para entrenamiento
img_train_gen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    vertical_flip=True,
    horizontal_flip=True
)

# Aumentar imágenes de entrenamiento
img_train_gen.fit(fotos_entrenamiento)

# Modelo de IA con capas convolucionales
modelo_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ancho, alto, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(etiquetas), activation='softmax')
])

# Compilar el modelo
modelo_cnn.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Callback para TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

# Entrenar el modelo
modelo_cnn.fit(img_train_gen.flow(fotos_entrenamiento, etiquetas_entrenamiento, batch_size=32),
               validation_data=(fotos_validacion, etiquetas_validacion),
               epochs=100, callbacks=[tensorboard_callback])

# Definir las rutas para guardar el modelo y los pesos
ruta_modelo = 'modelo_cnn.h5'
ruta_pesos = 'pesos_cnn.weights.h5'

# Guardar el modelo y los pesos
modelo_cnn.save(ruta_modelo)
modelo_cnn.save_weights(ruta_pesos)

print("Modelo y pesos guardados correctamente.")
