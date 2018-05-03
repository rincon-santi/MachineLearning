import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import math as math

def xtild(x):
    return np.concatenate(([[1]*X.shape[1]],X),axis=0)

# Calculates linear form to classify data using the least squares method
def least_squares(X,t):
    xtilde = xtild(X)
    Wtilde = np.dot(np.linalg.inv(np.dot(xtilde,xtilde.T)),np.dot(xtilde,t.T)) # we still need to check the determinant
    return Wtilde

# Calculates linear form to classify data using the Fisher (linear discriminant) method
def lda(X,t):
    mean_vectors = []
    for cl in [-1,1]:
        aux=X[:, np.where(t==cl)]
        aux.shape=(aux.shape[0],aux.shape[2])
        mean_vectors.append(np.mean(aux, axis=1))
    S_W = np.zeros((X.shape[0],X.shape[0]))
    clase=0
    var_class_mat=[]
    p=[]
    for cl,mv in zip([-1,1], mean_vectors):
        var_class_mat.append(np.zeros((X.shape[0],X.shape[0])))   
        aux=X[:, np.where(t==cl)]
        aux.shape=(aux.shape[0],aux.shape[2])
        p.append(float(aux.shape[1])/float(X.shape[1]))
        for i in (0, aux.shape[1]-1):
            col=aux[:,i]
            col.shape=(col.shape[0],1)
            mv.shape=(mv.shape[0],1)
            var_class_mat[clase] += np.dot((col-mv),(col-mv).T)
        S_W += var_class_mat[clase]
        clase+=1
    W=np.dot(np.linalg.inv(S_W),mean_vectors[1]-mean_vectors[0])  # we have to check determinant
    W.shape=W.shape[0]
    s=[]
    m=[]
    for i in (0, 1):
        s.append(np.dot(np.dot(W.T,var_class_mat[i]),W))
        m.append(np.dot(W.T,mean_vectors[i]))
    F=[(s[0]-s[1]),(2*(s[1]*m[0])),(-s[1]*pow(m[0],2)+s[0]*pow(m[1],2)-2*s[0]*s[1]*math.log(p[1]/math.sqrt(s[1]))+2*s[0]*s[1]*math.log(p[0]/math.sqrt(s[0])))]
    c=np.roots([F[0],F[1],F[2]])
    F_prima=[2*F[0], F[1]]
    if (math.copysign(c[0]*F_prima[0]-F_prima[1],1))<=(math.copysign(c[1]*F_prima[0]-F_prima[1],1)):
        wtilde=np.concatenate(([-(c[0]*pow(W[0],2)+c[0]*pow(W[1],2))], W.T)).T
    else:
        wtilde=np.concatenate(([-(c[1]*pow(W[0],2)+c[1]*pow(W[1],2))], W.T)).T
    return wtilde

# Classifies data X using linear form W
def classify(x,w):
    xtilde = np.concatenate(([1],x),axis=0)
    if len(w.shape) == 1:
        if (np.dot(xtilde,w) >= 0):
            return 1
        else:
            return -1
    else:
        return np.argmax(w.T.dot(np.array([1,a,b])))


#### GUI


class CreatePoints(object):
    
    def __init__(self, fig, ax):
        self.circle_list = []

        self.x0 = None
        self.y0 = None

        self.fig = fig
        self.ax = ax
        
        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.press_event = None
        self.current_circle = None
        self.color = 'r'

    def on_press(self, event):
        if ax != event.inaxes:
            return
        # First clear all lines
        for line in self.ax.lines:
            line.remove()
        
        x0, y0 = event.xdata, event.ydata

        
        for circle in self.circle_list:
            contains, attr = circle.contains(event)
            if contains:
                if event.button == 3:
                    self.circle_list.remove(circle)
                    circle.remove()
                    self.fig.canvas.draw()
                    return
                else:
                    self.press_event = event
                    self.current_circle = circle
                    self.x0, self.y0 = self.current_circle.center
                    self.fig.canvas.draw()
                    return

        c = Circle((x0, y0), 0.5, color=self.color)
        self.ax.add_patch(c)
        self.circle_list.append(c)
        self.current_circle = None
        self.fig.canvas.draw()

    def on_release(self, event):
        if ax != event.inaxes:
            return
        self.press_event = None
        self.current_circle = None

    def on_move(self, event):
        if ax != event.inaxes:
            return
        if (self.press_event is None or
            event.inaxes != self.press_event.inaxes or
            self.current_circle == None):
            return
        
        dx = event.xdata - self.press_event.xdata
        dy = event.ydata - self.press_event.ydata
        self.current_circle.center = self.x0 + dx, self.y0 + dy
        self.fig.canvas.draw()


fig, ax = plt.subplots()
ax.set_xlim([-20,20])
ax.set_ylim([-20,20])
plt.subplots_adjust(bottom=0.2)

circu = CreatePoints(fig,ax)

axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axslid = plt.axes([0.5, 0.05, 0.1, 0.075])
slid = Slider(axslid, 'Classes', 2, 5, valfmt='%d')

def update(val):
    pass

slid.on_changed(update)

axclass = plt.axes([0.3, 0.05, 0.1, 0.075])
bclass = Button(axclass, 'Class 1')
k = 1
def changeclass(event):
    global k
    global bclass
    k = k + 1
    bclass.label.set_text('Class {}'.format(k))
bclass.on_clicked(changeclass)

bnext = Button(axnext, 'Next')
#bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
#bprev.on_clicked(callback.prev)

ax.plot([1],[1],'o')
plt.show()
