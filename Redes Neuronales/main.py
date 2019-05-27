import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self, filename, n_neurons, n_outputs, test_size, activator = 'sigmoideal', shuffle = False):
        self.n_neurons          = n_neurons
        self.n_outputs           = n_outputs
        database                = pd.read_csv('../DataBase/' + filename, header = None)
        database = database.replace('Iris-setosa',0)
        database = database.replace('Iris-versicolor',1)
        database = database.replace('Iris-virginica',2)
        if shuffle == True:
            train, test = train_test_split(database.values, test_size = (test_size/100))
        else:
            database_length = len(database)
            test_size       = test_size / 100
            limit           = int(database_length * test_size)
            train, test = database.values[limit:], database.values[:limit]

        
        train, test   = np.insert(train, 0, 1,axis=1).astype('float64'), np.insert(test, 0, 1,axis=1).astype('float64')
        
        self.train_features = train[:,:-1] #Get all columns except the last
        self.train_targets  = train[:,-1]  #Get the last column

        self.test_features  = test[:,:-1]
        self.test_targets   = test[:,-1]

        self.features_size  = len(self.train_features[0])

        self.hidden_layer   = np.random.rand(self.features_size, self.n_neurons)
        self.output_layer   = np.random.rand(self.n_neurons + 1, self.n_outputs)

        if activator == 'sigmoideal':
            self.activator = lambda X: 1 / ( 1 + np.exp(-X))
        else:
            print("Not supported yet")
    
    def forward(self):
        hidden_net = self.train_features @ self.hidden_layer   #Calculate net to all inputs wich went to hidden layer

        hidden_output = self.activator(hidden_net)   #Apply Activation function
        hidden_output =  np.insert(hidden_output,0,1,axis=1)   #Add 1's columns

        output_net = hidden_output @ self.output_layer
        obtained_result = self.activator(output_net)
        print (obtained_result.shape)


        




        



mlp = MLP('shuffled_iris.data',3,2,20)
b = mlp.activator(np.array([1. , 5. , 3. , 1.6, 0.2]))
b = np.insert(b, 0, 1, axis=0)
mlp.forward()