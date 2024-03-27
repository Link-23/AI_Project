import tensorflow as tf
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array

ModeloCNN2 = r"modelo_cnn.h5"

CNN2= tf.keras.models.load_model(ModeloCNN2)
pesoscnn= CNN2.get_weights()
CNN2.set_weights(pesoscnn)

#Captura del video
cap = cv2.VideoCapture(0)

while True:
    #Lectura del video
    ret, frame = cap.read()
    
    #Se cambia a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Se redimensiona la imagen
    gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_CUBIC)
    
    #Normaliza la imagen
    gray = np.array(gray).astype(float)/255
    
    #Pasamos la imagen a matriz
    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)
    
    #Realiza la prediccion 
    prediction = CNN2.predict(img)[0][0]
    print(prediction)
    
    #Realizamos la clasificacion
    if prediction <= 0.5:
        cv2.putText(frame, 'Text', (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'Text', (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    
    #Mostramos los fotogramas 
    cv2.imshow('CNN', frame)
    
    t= cv2.waitKey(1)
    if t == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
    