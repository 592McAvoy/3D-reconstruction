import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time

from helper import *

def boundary_match(b1, b2, k):
    N1 = b1.shape[0]
    N2 = b2.shape[0]
    dp = []

    min = 9999
    tmp = []
    for i in range(N1):
        dist = distance(b2[0],b1[i])
        if(dist<min):
            min = dist
            tmp.append((min,i))
        else:
            tmp.append(tmp[-1])
    dp.append(tmp)
    
    for i in range(1,N2):
        tmp = []
        prev = dp[i-1]
        p = b2[i]
        tmp.append((9999,-1))
        for j in range(1,N1):
            d1 = tmp[j-1][0]
            prev = dp[i-1][j-1]
            if(j>prev[1] and j-prev[1]<=k):
                dist = distance(p, b1[j])
                d2 = dist + prev[0]
                if(d1<d2):
                    tmp.append(tmp[-1])
                else:
                    tmp.append((d2,j))
            else:
                tmp.append(tmp[-1])
        # print(tmp[-1])
        dp.append(tmp)

    match = []
    idx = N1-1
    for i in range(N2-1,-1,-1):
        idx = dp[i][idx][1]
        match = [idx]+match
        idx -= 1;   

    return match

def transfer(src_co, src_img, tgt_co):
    tgt_img = np.zeros(src_img.shape)
    tgt_img = 255 - tgt_img
    N = src_co.shape[0]
    for i in range(N):
        src = src_co[i]
        src = [to_closest(src[0]),to_closest(src[1])]
        tgt = tgt_co[i]        
        tgt_img[tgt[1]][tgt[0]] = src_img[src[1]][src[0]]
    return tgt_img

def test_img(img, co, name):
    out = np.zeros(img.shape)
    out = 255 - out
    for p in co:
        p = [to_closest(p[0]),to_closest(p[1])]
        out[p[1]][p[0]] = img[p[1]][p[0]]
    cv.imwrite(name,out)
 

