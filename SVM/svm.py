import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import make_confussion_matrix

database = pd.read_csv('../DataBase/test.data', header= None)

#Get Data and aplit fit_values from test_values
data_values = database.values[:,[0,1,2,3]].astype('float32')
targets = database.values[:,4]


targets [ targets == 'Iris-setosa' ] = -1
targets [ targets == 'Iris-versicolor' ] = 1
targets = targets.astype('int')


test_range = 60

data_test   = data_values [ : test_range]
target_test = targets [ : test_range]

verif_data		= data_values[test_range: ]
verif_target	= targets[test_range:]


#Choose features
features_indexes	= [0,1,2,3]
features_len		= len(features_indexes)
data_test			= data_test[:, features_indexes]

verif_data			= verif_data[:,features_indexes]

#SVM start
import cvxopt as co
#Initialize variables
K = target_test[:, None] * data_test
P = co.matrix(np.dot(K, K.T))
q = co.matrix(-np.ones((test_range, 1))) 
G = co.matrix(-np.eye(test_range)) 
h = co.matrix(np.zeros(test_range)) 

A = co.matrix(target_test.reshape(1, -1),tc='d')
b = co.matrix(np.zeros(1))

co.solvers.options['show_progress'] = False
sol = co.solvers.qp(P, q, G, h, A, b)

_lambda = np.array(sol['x'])
#print("lambda value: ", _lambda)


#Get w and b
n_w = sum ( [ _lambda[i] * target_test[i] * data_test[i]  for i in range(test_range) ] ) /test_range
n_b = sum ( [ -1 * _lambda[i] * target_test[i] * np.dot(data_test[i],data_test[i]) for i in range(test_range) ] ) / test_range
#n_b = list(sol['y'])[0]
#n_b = np.average(n_b)
print(n_w,n_b)


def get_class(x, w, b):
	return w @ x + b

#Run Verifications

verif_res = np.array([ get_class(verif_data[i], n_w, n_b)[0] for i in range(len(verif_data)) ])
verif_res [verif_res >= 0]	= 1
verif_res [verif_res < 0]	= -1
verif_res = verif_res.astype('int')

names = {-1:'Setosa',1:'Versicolor'}

mat = make_confussion_matrix(verif_target, verif_res, title = 'SVM', names =  names)
plt.show()





