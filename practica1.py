import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import math as math

def xtild(x):
    return np.concatenate(([[1]*x.shape[1]],x),axis=0)

colorcodes = {1 : 'r', 2 : 'g', 3 : 'b', 4 : 'c', 5 : 'y'}
codecolors = {'r' : 1, 'g' : 2, 'b' : 3, 'c' : 4, 'y' : 5}
codesigns = {1 : 1, 2 : -1}
signcodes = {1 : 1, -1 : 2}
def codeonehot(k, K):
    return np.array([0]*(k-1)+[1]+[0]*(K-k))

# This won't be necessary
#def onehotcode(arr):
#    if float(arr[0]) == 1.0

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
    global K
    xtilde = np.concatenate(([1],x),axis=0)
    if K == 2:
        if (np.dot(xtilde,w) >= 0):
            return 1
        else:
            return 2
    else:
        return np.argmax(w.T.dot(xtilde)) + 1


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
        self.color = colorcodes[1]

    def updateColor(self, k):
        self.color = colorcodes[k]

    def shrink(self, K):
        for circle, color in \
            [(circle,color) for (circle,color) in self.circle_list
             if codecolors[color] > K]:
            self.circle_list.remove((circle,color))
            circle.remove()
        self.fig.canvas.draw()

    def getX(self):
        if self.circle_list == []:
            return []
        x = np.array([[self.circle_list[0][0].center[0],
                       self.circle_list[0][0].center[1]]]).T
        for circle, color in self.circle_list[1:]:
            x = np.concatenate((x,
                            np.array([[circle.center[0], circle.center[1]]]).T
                                ),axis=1)
        return x
    
    def getSignT(self):
        return np.array([[1 if codecolors[color] == 1 else -1
                for circle,color in self.circle_list]])
        
    def getOneHotT(self, K):
        return np.array([ codeonehot(codecolors[color], K)
                for circle,color in self.circle_list]).T
        
    def on_press(self, event):
        if ax != event.inaxes:
            return
        # First clear all lines
        for line in [line for line in self.ax.lines]:
            line.remove()
        
        x0, y0 = event.xdata, event.ydata

        
        for circle,color in self.circle_list:
            contains, attr = circle.contains(event)
            if contains:
                if event.button == 3:
                    self.circle_list.remove((circle,color))
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
        self.circle_list.append((c,self.color))
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

def fsquares(event):
    x = circu.getX()
    t = circu.getSignT() if K == 2 else circu.getOneHotT(K)
    w = least_squares(x,t)
    grid = [(x,y) for x in np.linspace(-20,20,15) for y in np.linspace(-20,20,15)]
    for (a,b) in grid:
        clase = classify(np.array([a,b]), w)
        ax.plot(a,b,'.{}'.format(colorcodes[clase]))
    fig.canvas.draw()

def ffisher(event):
    pass


axminsquares = plt.axes([0.7, 0.05, 0.1, 0.075])
axfisher = plt.axes([0.81, 0.05, 0.1, 0.075])
bminsquares = Button(axminsquares, 'Squares')
bminsquares.on_clicked(fsquares)
bfisher = Button(axfisher, 'Fisher')
bfisher.on_clicked(ffisher)

axslid = plt.axes([0.5, 0.05, 0.1, 0.075])
slid = Slider(axslid, 'Classes', 2, 5, valfmt='%d')

K = 2
def update(val):
    global k
    global K
    global bfisher
    global bclass
    newK = int(math.floor(val))
    #Shrink data points and current class if necessary
    if newK < k:
        k = newK
        bclass.label.set_text('Class {}'.format(k))
    if newK < K:
        circu.shrink(newK)
    K = newK
    if K>2:
        # Hide Fisher button
        axfisher.patch.set_visible(False)
        bfisher.label.set_visible(False)
        axfisher.axis('off')
        fig.canvas.draw()
    else:
        # Show Fisher button
        axfisher.patch.set_visible(True)
        bfisher.label.set_visible(True)
        axfisher.axis('on')
        fig.canvas.draw()

slid.on_changed(update)


axclass = plt.axes([0.3, 0.05, 0.1, 0.075])
bclass = Button(axclass, 'Class 1')

k = 1
def changeclass(event):
    global k
    global bclass
    if k == K:
        k = 1
    else:
        k = k + 1
    circu.updateColor(k)
    bclass.label.set_text('Class {}'.format(k))
bclass.on_clicked(changeclass)


plt.show()
