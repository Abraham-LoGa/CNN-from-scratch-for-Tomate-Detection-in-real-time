import numpy as np
from numpy import asarray

class Convolution:
	
	def __init__(self, num_filters,filter_size):
		  # Se inicializan los parámetros para la convulución y creación de métodos
		self.num_filters = num_filters
		self.filter_size = filter_size
		self.conv_filter=np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)

	  # Función para la región de imagen a convolucionar
	def image_region(self, image):
		h = image.shape[0]
		w = image.shape[1]

		self.image = image
		for i in range(h-self.filter_size + 1):
			for j in range(w - self.filter_size + 1):
				image_patch = image[i : (i + self.filter_size), j : (j + self.filter_size)]
				yield image_patch, i, j 
	  # Propagación hacia adelante 
	def forward_prop(self, image):
		h,w = image.shape
		a = h - self.filter_size + 1
		b = w - self.filter_size + 1
		conv_out = np.zeros((a,b,self.num_filters))
		for image_patch,i,j in self.image_region(image):
			conv_out[i,j]=np.sum(image_patch*self.conv_filter,axis= (1,2))
		return conv_out 

	  # Propagación en retroceso dL_out es el gradiente de pérdida para las salidadeas de la capa
	def back_prop(self, dL_dout, learning_rate):
		dL_dF_params = np.zeros(self.conv_filter.shape)
		for image_patch, i, j in self.image_region(self.image):
			for k in range(self.num_filters):
				dL_dF_params[k] += image_patch*dL_dout[i,j,k]

		  # Actualización de filtros
		self.conv_filter -= learning_rate*dL_dF_params
		return dL_dF_params