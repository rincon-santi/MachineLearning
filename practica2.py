from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home = './data')
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

data = mnist.data

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim([0,28])
ax.set_ylim([28,0])
ax.set_aspect('equal')

#imagen = np.resize(data[27058], (28,28))
ax.add_patch(Rectangle((0,0),28,28,color='#000000'))
#ax.imshow(imagen, cmap='gray')
#plt.show()



value = 255
pressed = False

imagen = np.array([([0]*28)]*28)
#ax.imshow(imagen, cmap='gray')

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

axslid = plt.axes([0.5, 0.05, 0.1, 0.075])
slid = Slider(axslid, 'Brightness', 0, 255, valfmt='%d', valinit=255)

def update(val):
    global value
    value = int(math.floor(val))
    # prueba jincha
    ax.clear()
    ax.imshow(imagen, cmap='gray')


slid.on_changed(update)


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()
