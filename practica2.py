from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

mnist = fetch_mldata('MNIST original', data_home='./data')

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim([0, 28])
ax.set_ylim([28, 0])
ax.set_aspect('equal')

ax.add_patch(Rectangle((0,0),28,28,color='#000000'))

value = 255
pressed = False

imagen = np.array([([0]*28)]*28)


def dibuja(event):
    global imagen
    x = int(math.floor(event.xdata))
    y = int(math.floor(event.ydata))
    color = '#'+ ("%02x" % value)* 3
    ax.add_patch(Rectangle((x,y),1,1,color=color))
    imagen[y][x] = value
    fig.canvas.draw()    

def on_press(event):
    global pressed
    if event.inaxes != ax:
        return
    dibuja(event)
    pressed = True
def on_release(event):
    global pressed
    pressed = False
def on_move(event): # event.inaxes
    if  pressed and event.inaxes == ax:
#        print("{}{}".format(event.xdata,event.ydata))
        dibuja(event)    

axclear = plt.axes([0.8, 0.05, 0.1, 0.075])
bclear = Button(axclear, 'Clear')

axslid = plt.axes([0.5, 0.05, 0.1, 0.075])
slid = Slider(axslid, 'Brightness', 0, 255, valfmt='%d', valinit=255)

def update(val):
    global value
    value = int(math.floor(val))

def clearcanv(event):
    global imagen
    [p.remove() for p in reversed(ax.patches)]
    ax.add_patch(Rectangle((0,0),28,28,color='#000000'))
    imagen = np.array([([0]*28)]*28)
    print('Clear')

fig.canvas.set_window_title('Practica 2.a')

bclear.on_clicked(clearcanv)
slid.on_changed(update)

axclsfy = plt.axes([0.2, 0.05, 0.1, 0.075])
bclsfy = Button(axclsfy, 'Classify')


#### Perceptron

class Perceptron:
    def __init__(self, D):
        self.w = np.array([0.0]*(D+1))
    
    def eval_weights(self, x):
        return np.dot(self.w, np.concatenate(([1],x)))
    
    def eval(self, x):
        return 1 if self.eval_weights(x) > 0 else -1
    
    def train(self, X, T, w0 = None, eta = 0.1):
        if w0 != None:
            self.w = w0
        Xtilde = np.concatenate(([[1] for i in range(X.shape[0])],X), axis=1)
        for i in np.random.permutation(range(len(X))): # np.random.randint(0, len(X), size=len(X)):
            if T[i] * np.dot(Xtilde[i],self.w) <= 0:
                self.w = self.w + eta * T[i] * Xtilde[i] 
    
    def get_weights(self):
        return self.w

#### Training data and targets

TAMUESTRA = 0.8

muestra = np.random.permutation(range(len(mnist.data)))
tamanamiento = int(math.floor(TAMUESTRA * len(mnist.data)))

data = mnist.data[muestra][:tamanamiento]
target = mnist.target[muestra][:tamanamiento]

perceptrones = [Perceptron(data.shape[1]) for i in range(10)]
for i in range(10):
    perceptrones[i].train(data, [1 if t == i else 0 for t in target])

def clsfy(x):
    return np.argmax([perceptrones[i].eval_weights(x) for i in range(10)])

#### Test data and targets
tdata = mnist.data[muestra][tamanamiento:]
ttarget = mnist.target[muestra][tamanamiento:]

correctos = 0
for i in range(len(tdata)):
    if clsfy(tdata[i]) == ttarget[i]:
        correctos = correctos + 1
print( 'Tamaño del training set: {}. Tamaño del test set: {}. Aciertos: {}'.format(
    tamanamiento, len(mnist.data) - tamanamiento, correctos) )

def fclsfy(event):
    print( clsfy(np.resize(imagen,28*28)) )

bclsfy.on_clicked(fclsfy)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()
