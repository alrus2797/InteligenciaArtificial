import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def construct_class_vector(id, n):
	a = np.zeros(n)
	a[int(id)] = 1
	return a

def construct_target_vectors(vector, n_classes):
	res = []
	for i in vector:
		res.append(construct_class_vector(i,n_classes))
	return np.array(res)
	


class MLP:
	def __init__(self, filename, classes, n_neurons, n_outputs, alpha, test_size, activator = 'sigmoideal', shuffle = False):
		self.n_neurons          = n_neurons
		self.n_outputs			= n_outputs
		self.alpha				= alpha
		database                = pd.read_csv('../DataBase/' + filename, header = None)

		for class_id, class_name in classes.items():
			database = database.replace(class_name, class_id)
		n_classes = len(classes)
		
		if shuffle == True:
			train, test = train_test_split(database.values, test_size = (test_size/100))
		else:
			database_length = len(database)
			test_size       = test_size / 100
			limit           = database_length - int(database_length * test_size)
			train, test = database.values[:limit], database.values[limit:]

		
		train, test   = np.insert(train, 0, 1,axis=1).astype('float64'), np.insert(test, 0, 1,axis=1).astype('float64')
		
		self.train_features = train[:,:-1] #Get all columns except the last
		train_targets  = train[:,-1]  #Get the last column

		self.test_features  = test[:,:-1]
		test_targets   = test[:,-1]

		#Convert classes to vectors

		self.train_targets = construct_target_vectors(train_targets, n_classes)
		self.test_targets = construct_target_vectors(test_targets, n_classes)
		

		self.features_size  = len(self.train_features[0])

		self.hidden_layer   = np.random.rand(self.features_size, self.n_neurons)
		self.output_layer	= np.random.rand(self.n_neurons + 1, self.n_outputs)

		if activator == 'sigmoideal':
			self.activator = lambda X: 1 / ( 1 + np.exp(-X))
		else:
			print("Not supported yet")
	
	def forward(self):
		hidden_net = self.train_features @ self.hidden_layer   #Calculate net to all inputs wich went to hidden layer

		hidden_output = self.activator(hidden_net)   #Apply Activation function
		self.hidden_output =  np.insert(hidden_output,0,1,axis=1)   #Add 1's columns

		output_net = self.hidden_output @ self.output_layer
		self.obtained_result = self.activator(output_net)
		#print (self.obtained_result.shape, self.hidden_output.shape)

	def backward(self):
		first_block_chain	= self.obtained_result - self.train_targets
		second_block_chain	= self.obtained_result * (1 - self.obtained_result) # To make a dot product
		third_block_chain	= self.hidden_output

		delta	= first_block_chain * second_block_chain	
		self.delta = delta
		chain	= (delta.T @ self.hidden_output).T	#Chain Rule derivative result

		#print("old\n", self.output_layer)
		self.output_layer -= self.alpha * (chain)
		#print("new\n", self.output_layer)

		#second_part = (delta @  self.output_layer.T).T @  (self.hidden_output * (1 - self.hidden_output) * self.train_features[:,1:])
		second_part = ((delta * ((self.hidden_output * (1 - self.hidden_output)) @ self.output_layer)).T @ self.train_features).T
		self.hidden_layer -= self.alpha * second_part
	def get_error(self):
		return ((self.obtained_result - self.train_targets)**2)/2

	def fit(self, iterations):
		for i in range(iterations):
			self.forward()
			self.backward()
			#if i %1000 ==0 : print(self.get_error())
		print (self.hidden_layer, self.output_layer)


classes = {
	0: 'Iris-setosa',
	1: 'Iris-versicolor',
	2: 'Iris-virginica',
}
mlp = MLP('shuffled_iris.data', classes, 3, 3, 0.1,20)

mlp.fit(100000)
# b = mlp.activator(np.array([1. , 5. , 3. , 1.6, 0.2]))
# b = np.insert(b, 0, 1, axis=0)
# mlp.forward()
# mlp.backward()