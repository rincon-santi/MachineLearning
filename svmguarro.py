import numpy as np

x = np.concatenate((np.random.rand(2,15) * 20,  np.random.rand(2,15) * -20), axis=1)
t = np.array([1]*15 + [-1]*15)
w = np.zeros((2,1))
a = np.zeros(t.shape[0])

# kernel asunting

# k = x.T @ x

# def kern(u,v):
#     return np.dot(u,v).squeeze()

# next kerneling

k = np.polynomial.polynomial.polyval( x.T @ x, [1,1,2] )
def kern(u,v):
    return np.polynomial.polynomial.polyval( np.dot(u,v).squeeze(), [1,1,2] )


b = 0

C = 10

def E_unbiased(i):
    return t @ (a * k[:,i] ) - t[i]

def a_j_super1(i,j):
    return a[j] + (float(t[j]*(E_unbiased(i)-E_unbiased(j)))/
                   float(k[i,i] + k[j,j] - 2 * k[i,j]))

def L(i,j):
    return max(0, a[j] - a[i]) if t[i] != t[j] else max(0, a[j] + a[i] - C)

def H(i,j):
    return min(C, C + a[j] - a[i]) if t[i] != t[j] else min(C, a[j] + a[i])

def computeStep(i,j):
    vH = H(i,j)
    vL = L(i,j)
    va = a_j_super1(i,j)
    na = vH if va >= vH else va if va > vL else vL
    a[i] = a[i] + t[i]*t[j]*(a[j] - na)
    a[j] = na

def updateBias():
    global b
    global S
    S = np.where(a)[0]
    M = np.intersect1d(S, np.where(C-a))
    if len(M) > 0:
        b = 1.0/len(M) * np.sum( (t - np.sum( ((a*t)[:,np.newaxis]*k)[S], axis=0 ))[M] )

def wpor(xx):
    return np.sum(a*t*kern(x.T, xx[:,np.newaxis]))



def train(aa):
    if aa == None:
        a = np.zeros(t.shape[0])
    for i in range(100000):
        u = np.random.randint(len(x.T))
        v = np.random.randint(len(x.T))
        while v == u:
            v = np.random.randint(len(x.T))
        computeStep(u,v)
    updateBias()

import matplotlib.pyplot as plt

def draw():
    plt.cla()
    poin = np.array([[a,b] for a in np.linspace(-20,20,15) for b in np.linspace(-20,20,15)])
    wpoin = np.array([wpor(i) + b for i in poin])

    plt.plot(poin[wpoin > 0,0], poin[wpoin > 0, 1], 'r.')

    plt.plot(poin[wpoin < 0,0], poin[wpoin < 0, 1], 'b.')

    rrr = np.intersect1d(np.where(a), np.arange(0,15))
    bbb = np.intersect1d(np.where(a), np.arange(15,30))
    plt.plot(x[:,rrr][0],x[:,rrr][1],'Dr')
    plt.plot(x[:,bbb][0],x[:,bbb][1],'Db')
    rr = np.intersect1d(np.where(a == 0), np.arange(0,15))
    bb = np.intersect1d(np.where(a == 0), np.arange(15,30))
    plt.plot(x[:, rr][0],x[:, rr][1],'or')
    plt.plot(x[:,bb][0],x[:,bb][1],'ob')

    # print(x.T[np.where(a)])


    plt.show()


if __name__=="__main__":
    train(None)
    draw()
