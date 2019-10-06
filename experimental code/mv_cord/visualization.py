import numpy as np
import matplotlib.pyplot as plt


def draw_points(pts, r):
    fig = plt.figure()
    axe = fig.add_subplot(111)
    X,Y = [],[]
    fig.show()
    for point in pts:
        X.append(point[0])
        Y.append(point[1])
        axe.cla()
        axe.plot(X,Y, 'ro')
        axe.set_xlim(0,r)
        axe.set_ylim(r,0)
        fig.canvas.draw()

def draw_matching(b1, b2, r):
    L = b1.shape[0]
    
    fig = plt.figure()
    axe = fig.add_subplot(111)
    X,Y = [],[]
    X1,Y1 = [],[]
    fig.show()
    for i in range(L):
        p1 = b1[i]
        p2 = b2[i]
        X.append(p1[0])
        Y.append(p1[1])
        axe.cla()
        axe.plot(X,Y, 'ro')
        axe.set_xlim(0,r)
        axe.set_ylim(r,0)
        X1.append(p2[0])
        Y1.append(p2[1])
        axe.plot(X1,Y1, 'bo')
        axe.set_xlim(0,r)
        axe.set_ylim(r,0)
        fig.canvas.draw()
        