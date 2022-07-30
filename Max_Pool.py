import numpy as np
from numpy import asarray

class Max_Pool:

	def __init__(self, filter_size):
		self.filter_size = filter_size 
	
	  # Función para generar regiones de imágenes no superpuestas para la agrupación
	def image_region(self, image):
		new_h = image.shape[0]//self.filter_size
		new_w = image.shape[1]//self.filter_size
		self.image = image
		for i in range(new_h):
			for j in range(new_w):
				a = i*self.filter_size
				b = i*self.filter_size + self.filter_size
				c = j*self.filter_size
				d = j*self.filter_size + self.filter_size
				image_patch = image[a:b,c:d]
				yield image_patch, i, j
	
	  # Función que realiza una pasada hacia delante utilizando las entradas dadas
	  # Devuelve una matriz de 3D
	def forward_prop(self, image):
		height, widht, num_filters = image.shape
		output = np.zeros((height//self.filter_size, widht//self.filter_size, num_filters))
		
		for image_patch, i, j in self.image_region(image):
			output[i,j] = np.amax(image_patch, axis = (0,1))

		return output 

	  # Realiza el pasao hacia atras devolviendo el degradado de pérdida para las entrafas de esta capa
	def back_prop(self,dL_dout):
		dL_dmax_pool = np.zeros(self.image.shape)
		for image_patch, i, j in self.image_region(self.image):
			h,w,num_filters = image_patch.shape
			maximun_val = np.amax(image_patch, axis = (0,1))

			for x in range(h):
				for y in range(w):
					for z in range(num_filters):
						if image_patch[x,y,z] == maximun_val[z]:
							dL_dmax_pool[i*self.filter_size + x, j*self.filter_size +y,z]=dL_dout[i,j,z]
			return dL_dmax_pool
