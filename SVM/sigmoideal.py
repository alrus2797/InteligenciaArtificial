import pandas as pd

from sklearn.model_selection import train_test_split
import math

import matplotlib.pyplot as plt

import numpy as np

from utils import make_confussion_matrix


class Reg:
    def __init__(self,X,Y,alpha=0.07,umbral=0.01):
        self.X = X
        self.Y = Y
        self.thetas = np.array([], dtype = 'float32')
        self.alpha  = alpha
        self.umbral = umbral

        self.X = np.insert(X,0,1,axis=1)
    
    def h(self,x):
        #print (x, self.thetas)
        return x @ self.thetas.T
    
    def s(self,x):
        return 1 / (1 + np.exp(-self.h(x)) )

    def get_error(self):
        return -1 * sum([ self.Y[i] * math.log(self.s(self.X[i])) + (1 - self.Y[i]) * math.log(1 - self.s(self.X[i])) for i in range(len(self.X))])/len(self.X)
    
    def update_thetas(self):
        copy = self.thetas.copy()
        for j in range(len(self.thetas)):
            copy[j] -= self.alpha * sum( [ (self.s(self.X[i]) - self.Y[i]) * self.X[i][j] for i in range(len(self.X))] ) / len(self.X)
        self.thetas = copy
    def fit(self):
        self.thetas = np.random.rand(1,5)
        self.error =  self.get_error()
        old_error = 0
        #print ('fit',self.thetas)
        while (abs(old_error - self.error) >= 0.00000001):
        #while (self.error >= self.umbral):
            self.update_thetas()
            old_error = self.error
            self.error = self.get_error()
            print(self.thetas, self.error)
        print("error",self.error)
    
    def test(self, test, targets):
        test_result = [ self.s(test[i])[0] for i in range(len(test)) ]
        #print('test-result',test_result)
        return test_result

        



database = pd.read_csv('../DataBase/test.data',header=None)

data_values = database.values[:,[0,1,2,3]].astype('float32')
targets = database.values[:,4]

targets [ targets == 'Iris-setosa' ] = 0
targets [ targets == 'Iris-versicolor' ] = 1
targets = targets.astype('int')


test_range = 60

data_test = data_values [ : test_range]
target_test = targets [ : test_range]

verif_data      = data_values[test_range: ]
verif_target    = targets[test_range:]

reg = Reg(data_test,target_test,umbral = 0.6,alpha = 0.1)
reg.fit()

print ("predict",reg.s([1,5.4,3.4,1.5,0.4]))

verif_data = np.insert(verif_data,0,1,axis=1)

verif_res = reg.test(verif_data, verif_target)
#print(verif_res)

verif_res = np.array(verif_res)

verif_res[ verif_res < 0.5 ] = 0
verif_res[ verif_res >= 0.5 ] = 1
verif_res = verif_res.astype('int')

names = {0:'Setosa',1:'Versicolor'}

make_confussion_matrix(verif_target, verif_res, title = 'Logistic Regression', names = names)
plt.show()