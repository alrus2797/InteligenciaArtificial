import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from utils import make_confussion_matrix

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
		self.test_targets_as_ints	= test_targets.astype('int64')

		self.train_targets			= construct_target_vectors(train_targets, n_classes)
		self.test_targets			= construct_target_vectors(test_targets, n_classes)
		

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
		#print (self.hidden_layer, '\n' ,self.output_layer)

	def evaluate(self, input_features):
		#print("in",input_features)
		hidden_net = input_features @ self.hidden_layer
		hidden_output = self.activator(hidden_net)
		hidden_output = np.insert(hidden_output, 0, 1, axis = 1)
		output_net = hidden_output @ self.output_layer
		obtained_result = self.activator(output_net)
		# print(obtained_result)
		return obtained_result

	def test(self):
		evaluation	= self.evaluate(self.test_features)
		maximums	= np.amax(evaluation,axis=1)[None].T	#Get max values for each row and cast as (n,1) matrix
		indexes		= np.where(evaluation == maximums)[1]

		self.indexes	= indexes
		diff	= indexes == self.test_targets_as_ints
		success	= diff.sum()	#Count Trues
		failed	= len(diff) - success
		
		print("success:\t", success)
		print("failed: \t", failed)
		#print(selfle.indexes)

	def make_confussion_matrix(self, save=True, names = {}, title = 'Confussion Matrix', filename='temp', folder='Images'):
		if hasattr(self, 'indexes'):
			#fig = plt.gcf()
			plt.rcParams["figure.figsize"] = (7,7)
			cf_matrix	= make_confussion_matrix(self.test_targets_as_ints, self.indexes, title = title, names = names)
			if save:
				plt.savefig(folder+'/'+filename+'.png',dpi=300)
				#plt.close(fig)
			#plt.show()
		else:
			print('Not indexes defined. Run fit() function first')
		return cf_matrix





classes = {
	0: 'Iris-setosa',
	1: 'Iris-versicolor',
	2: 'Iris-virginica',
}

n_neurons	= [4,6,8,10,12]
alphas		= [0.01,0.04,0.07,0.1,0.5]


for neuron in n_neurons:
	for alpha in alphas:
		mlp = MLP('shuffled_iris.data', classes, 3, 3, 0.1,30)
		np.set_printoptions(precision=5, suppress=True)
		mlp.fit(100000)
		print('-------------------------')
		print("Test - neurons:",neuron,"alpha:", alpha )
		mlp.test()
		cf_matrix	= mlp.make_confussion_matrix(names=classes, title='', filename='test_'+str(neuron)+'_'+str(alpha), folder='Images')
		print("Score:\t\t", np.trace(cf_matrix))