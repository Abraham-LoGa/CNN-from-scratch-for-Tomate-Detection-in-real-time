  # Importamos librerias
import cv2
import numpy as np
import imutils
import os
  # Nombre de la carpeta donde se guardarán las imágenes
path_data = 'negative'
  # Condicional por si la carpeta existe 
if not os.path.exists(path_data):
    print('Carpeta creada: ',path_data)
    os.makedirs(path_data)

  # Inicialización de video
data = cv2.VideoCapture(2,cv2.CAP_DSHOW)
  # Tamaño de imagen
x1, y1 = 170, 89
x2, y2 = 470, 389

  # Número de imagen
count = 0
while True:
      # Leemos el frame y ret
    ret, frame = data.read()
      # Condicional que rompe el ciclo si no ret no es real 
    if ret == False: break
      # Mostramos los objetos dentro del parámetro del tamaño de imagen
    im_frame = frame.copy()
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
      # Capturamos el frame y lo guardamos a un tamaño de 38x38
    object_ = im_frame[y1:y2,x1:x2]
    object_ = imutils.resize(object_,width=38)


    k = cv2.waitKey(1)
      # Capturamos y guardamos la imagen en la dirección de la carpeta
    if k == ord('s'):
        cv2.imwrite(path_data+'/not_{}.jpg'.format(count),object_)
        print('Image save:'+'/not_{}.jpg'.format(count))
        count = count +1
      # Rompe el ciclo para parar el programa.
    if k == 27:
        break

    cv2.imshow('frame',frame)
    cv2.imshow('objeto',object_)

data.release()
cv2.destroyAllWindows()