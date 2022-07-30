"""
CNN from Scratch for Tomato Detection
This project was made by: Abraham LoGa
If you have any questions, please contact me: abraham.lg.chap@gmail.com
"""
  # Importamos librerías
import cv2
import os
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
  # Llamamos los archivos para entrenamiento de la CNN
from Convolution import Convolution
from Max_Pool import Max_Pool
from Softmax import Softmax


# Función para el redimensionamiento y la obtención de valores de imágenes 
def make_data(path,h,w):
	data_set = []  # Matriz donde se guardaran los valores de imágenes
	label = ["positive","negative"]  # Nombre de las carpetas
	  # Ciclo para la extracción de información de cada carpeta
	for i in label:
		c_num = label.index(i)  # Nombre de cada archivo
		d_path = os.path.join(path,i)  # Obtención de cada imagen de la carpeta
		  # Ciclo para cada una de las imágenes dentro de la carpeta
		for file in os.listdir(d_path):
			img = cv2.imread(os.path.join(d_path,file), cv2.IMREAD_GRAYSCALE)  # Lectura y conversión a grises de cada imagen
			img = cv2.resize(img,(h,w))  # Redimensionamiento de imágenes
			data_set.append([img,c_num]) # Se añade cada imagen e índice a la matriz vacía
	  # Declaración de nuevas matrices
	Data_mat=[] 
	Data_name=[]
	  # Ciclo para guardar los valores obtenidos anteriormente
	for info, label in data_set:
		  # Se añaden los valores
		Data_mat.append(info)
		Data_name.append(label)
	  # Se guardan los valores
	Data_mat = np.array(Data_mat).reshape(-1,h,w)
	Data_name = np.array(Data_name)
	return Data_mat,Data_name

Matriz, Label=make_data("Data_base",150,150)

train_images = Matriz[:210] # Imágenes de entramiento
train_labels = Label[:210]

Convolution_ = Convolution(8,3)                  # 8 filtros con tamaño de 3
M_pool = Max_Pool(2)               # Tamaño de filtro: 2
Softmax_ = Softmax(74*74*8, 10)


def forward(image, label):
 
   # Transformación de la imagen para su fácil manejo
  out = Convolution_.forward_prop((image / 255) - 0.5)
  out = M_pool.forward_prop(out)
  out = Softmax_.forward_prop(out)

  # Calcula la pérdida de entropía cruzada y precisión
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

  # Función para el entrenamiento completo de la imagen y etiqueta dada
def train(im, label, lr=.005):
  
  out, loss, acc = forward(im, label)

  # Calcula el gradoemte inicial
  gradient = np.zeros(2)
  gradient[label] = -1 / out[label]

  # Paso hacia atrás del gradiente
  gradient = Softmax_.back_prop(gradient, lr)
  gradient = M_pool.back_prop(gradient)
  gradient = Convolution_.back_prop(gradient, lr)

  return loss, acc

for epoca in range(1):

  # Se permuta los datos de entrenamiento
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  a=0
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):

    if i % 100 == 99:
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc
    if a%4==0:
    	print("CARGANDO .............................")
    a=a+1
# Test the CNN
print('\nInicio de Detección de Tomate')
loss = 0
correct = 0
n=0

cap = cv2.VideoCapture(0)
data_frame = []

while True:
  ret,frame= cap.read()
  img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  cv2.imshow("DETECTANDO",frame)
  img_detection=cv2.resize(img,(150,150))
  X = (img_detection)
  X=np.array(X).reshape(-1,150,150)
  Y = np.array([0])
  test_images = X[:1]
  test_labels = Y[:1]

  for im, label in zip(test_images,test_labels):
  	_, l, acc = forward(im, label)
  	loss += l
  	correct += acc

  	if acc == 1:
  		print("-------------- TOMATE DETECTADO -------")
  	else:
  		print("SIN TOMATE")
  k=cv2.waitKey(5)
  if k==27:
  	break

cv2.destroyAllWindows()

