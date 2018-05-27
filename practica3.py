# Nube de N puntos X \in \R^D, proyeccion afin a \R^\Dp minimo cuadratica
# Implementamos PCA mediante SVD
## Primer punto. Proyectar X usando PCA, determinar \Dp en funcion de \eps como se ha visto en clase (?)
# Implementamos LDA multiclase y clasificador bayesiano
## Segundo punto. Misma cuestion sobre \Dp en funcion de \eps (?). Las nubes de puntos las generamos con gaussianas multivariantes.
class LDA_classifier:
    def __init__(self, X, T):
        # init
        return
    def train(epsilon):
        #Proyecta y entrena un clasificador bayesiano
        return
    def classify(x):
        #no tengo claro aun que pasara si no se ha entrenado previamente
        return
# LDA contra MNIST, 80 training 20 testing. Tenemos que importar las practicas anteriores.
# https://scikit-learn.org/stable/datasets/index.html
# https://en.wikipedia.org/wiki/MNIST_database
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home = './data')
# Breast Cancer Wisconsin Diagnostic dataset. D = 30, N = 596, K = 2
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
## Hacer PCA a un subespacio obtimo e interpretar
## LDA. Comparar con practicas anteriores. Discutir que atributos caracterizan los tumores malignos.
from sklearn.datasets import load_breast_cancer
bcwd = load_breast_cancer() #sera necesaria codificacion one-hot



