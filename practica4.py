import numpy as np

sigmoid = lambda x : 1.0 / (1 + np.exp(-x))
sigmoidprima = lambda x : sigmoid(x) * ( 1 - sigmoid(x) ) 
softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0)

class multilayer_perceptron:
    def __init__(self, dimentions):
        self.L = len(dimentions) -1
        self.dim = dimentions
        self.w = [None] +[np.zeros((dimentions[i],dimentions[i-1]))
                          for i in np.arange(1,len(dimentions))]
        self.b = [None] + [np.zeros(dimentions[i])
                          for i in np.arange(1,len(dimentions))]

    def train(self, X, T, eta=0.1):
        indexes = np.random.permutation(X.shape[1])
        for idx in indexes:
            # Prealimentacion
            z = [np.zeros(i) for i in self.dim]
            a = [np.zeros(i) for i in self.dim]
            z[0] = X[:,idx]
            for i in range(self.L):
                a[i+1] = self.w[i+1] @ z[i] + self.b[i+1]
                z[i+1] = sigmoid(a[i+1])
            # Calculo de errores
            err = [None] + [np.zeros(self.dim[i])
                            for i in np.arange(1,len(self.dim)) ]
            err[self.L] = z[self.L]-T[:,idx]
            for i in np.arange(self.L-1, 0, -1):
                err[i] = np.diag(sigmoidprima(a[i])) @ self.w[i+1].T @ err[i+1]
            # Calculo del gradiente
            for k in np.arange(1,self.L+1):
                self.w[k] = self.w[k] - eta * (err[k][:,np.newaxis] @ z[k-1][np.newaxis,:])
                self.b[k] = self.b[k] - eta * err[k]


    def classify(self, x):
        z = [np.zeros(i) for i in self.dim]
        z[0] = x
        for i in range(self.L):
            z[i+1] = sigmoid(self.w[i+1] @ z[i] + self.b[i+1])
        return softmax(z[self.L])


# aqui empieza lo guarro

x = np.concatenate((np.random.rand(2,15) * 20,  np.random.rand(2,15) * -20), axis=1)
t = np.concatenate(([[1]*15,[0]*15],  [[0]*15,[1]*15]), axis=1)

perc = multilayer_perceptron([2,3,2])
perc.train(x,t)
