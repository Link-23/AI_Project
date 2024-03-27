#Este archivo es una versión preliminar de la IA, se realizó con fines de aprendizaje 
# Importar librerias
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Imagenes
#Se ingresan las rutas en las que tengas los datos para entrenar a la ia
entrenamiento = r"Ingresar la ruta de la capetas con los archvos para entrenar la ia"
validacion = r"Ingresar la ruta de la carpeta con los archivos para la Validacion de la ia"

listaTrain =os.listdir(entrenamiento)
listaTest =os.listdir(validacion)

#parametros
ancho, alto=200,200
#Lista de entrenamiento
etiquetas =[]
fotos=[]
datos_train=[]
con=0
#Listas de validacion
etiquetas2=[]
fotos2=[]
datos_vali=[]
con2=0

#Extraer la lista de fotos y entra los datos a las etiquetas
##Entrnamiento
for nameDir in listaTrain:
    nombre=entrenamiento+'/'+nameDir #Se lee la foto
    
    for fileName in os.listdir(nombre): #se le adigna una etiqueta a cada foto
        etiquetas.append(con) #Valor de la etiqueta (Asignamos 0 a la primera etiqueta y 1 a la segunda)
        img = cv2.imread(nombre + '/' + fileName, 0)  # Se lee la imagen
        img = cv2.resize(img,(ancho,alto),interpolation=cv2.INTER_CUBIC) #Se redimensiona la imagen
        img = img.reshape(ancho,alto,1) #Dejamos un solo canal
        datos_train.append([img,con])
        fotos.append(img) #Se agregan las imagenes en EDG
        
    con+=1
    
#Validacion 
for nameDir2 in listaTest:
    nombre2=validacion+'/'+nameDir2 #Se lee la foto
    
    for fileName2 in os.listdir(nombre2): #se le adigna una etiqueta a cada foto
        etiquetas2.append(con2)
        img2 = cv2.imread(nombre2+'/'+fileName2,0) # Se lee la imagen
        img2 = cv2.resize(img2,(ancho,alto),interpolation=cv2.INTER_CUBIC) #Se redimensiona la imagen
        img2 = img2.reshape(ancho,alto,1) #Dejamos un solo canal
        datos_vali.append([img2,con2])
        fotos2.append(img2) #Se agregan las imagenes en EDG
        
    con2+=1
    
#Normalizar las imagenes (0 o 1)
fotos =np.array(fotos).astype(float)/255
print(fotos.shape)
fotos2 =np.array(fotos2).astype(float)/255
print(fotos2.shape)
#Se pasan las listas a array
etiquetas=np.array(etiquetas)
etiquetas2=np.array(etiquetas2)

imgtrainGen=ImageDataGenerator(
    #rescale = 1./255
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=15,
    zoom_range=[0.5,1.5],
    vertical_flip=True,
    horizontal_flip=True
)

imgtrainGen.fit(fotos)
plt.figure(figsize=(20,8))

for imagen, etiqueta in imgtrainGen.flow(fotos, etiquetas, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i], cmap='gray')
    plt.show()
    break

imgTrain = imgtrainGen.flow(fotos, etiquetas, batch_size=32)

#Modelo de IA con capas densas

#Modelo de IA con capas convolucionales y drop out
ModeloCNN2=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)), #Capa de entrada convolucional con 32kernel
    tf.keras.layers.MaxPooling2D(2, 2), #Capa con maxpooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), #Capa de entrada convolucional con 64kernel
    tf.keras.layers.MaxPooling2D(2, 2), #Capa con maxpooling
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), #Capa de entrada convolucional con 128kernel
    tf.keras.layers.MaxPooling2D(2, 2), #Capa con maxpooling
    
    #Capas densas clasificacion
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Se compila los modelos: se agregan el optimizador y la funcion de perdida

ModeloCNN2.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


#Se observa y entrena las redes
#Para visualzar: tenserboard --logdir=r"logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

#Entrenamos Convoluvional con dropout
ModeloCNN2.fit(imgTrain, batch_size=32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [tensorboard_callback], steps_per_epoch = int(np.ceil(len(fotos)/float(32))),
                validation_steps = int(np.ceil(len(fotos2)/float(32))))
#Se guarda el modelo
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('pesoCNN2.weights.h5')
print('Ternimo del modelo CNN2')
