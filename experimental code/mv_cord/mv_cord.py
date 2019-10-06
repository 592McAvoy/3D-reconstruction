import numpy as np
import cv2 as cv
import math
from . import helper as h

def det(p1,p2):
    return p1[0]*p2[1] - p1[1]*p2[0]
def dot_prod(p1,p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

def F(p, verts):
    N = verts.shape[0]

    S = [0]*N
    for i in range(N):
        v = verts[i]
        S[i] = v-p
    
    R = [0]*N
    A = [0]*N
    D = [0]*N
    ret = [0]*N
    for i in range(N):
        i_1 = (i+1)%N
        R[i] = h.distance(S[i], [0,0])
        A[i] = det(S[i],S[i_1])*0.5
        D[i] = dot_prod(S[i],S[i_1])
        if R[i] == 0:            
            ret[i]=1
            ret = np.array(ret)
            return ret
        if A[i] == 0 and D[i]<0:
            R[i_1] = h.distance(S[i_1], [0,0])
            ret[i] = 1.0/(R[i]+R[i_1])
            ret[i_1] = R[i]*1.0/(R[i]+R[i_1])
            ret = np.array(ret)
            return ret

    for i in range(N):
        ip1 = (i+1)%N
        im1 = i-1
        w = 0
        if A[im1] != 0:
            w = w + (R[im1]-D[im1]*1.0/R[i])*1.0/A[im1]
        if A[i] != 0:
            w = w + (R[ip1]-D[i]*1.0/R[i])*1.0/A[i]
        ret[i] = w

    ret = np.array(ret)
    s = np.sum(ret)
    ret = ret/s
    return ret


def get_mv_coordinate(in_pts, b_pts):
    M = in_pts.shape[0] # M,2
    N = b_pts.shape[0]  # N,2
    cord = []
    dp = 0
    for pt in in_pts:
        w = F(pt, b_pts)
        #pp = np.dot(w, b_pts)
        #dp += distance(pt, pp1)
        cord.append(w)
    
    cord = np.array(cord) #shape:M,N
    #print("mead dp:",dp/M)
    print("cord shape",cord.shape)
    return cord

    
    

        


