import numpy as np
import cv2 as cv
import math


def distance(p1, p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow((p1[1]-p2[1]),2))

def to_closest(x):
    i = math.floor(x)
    if x-i<0.5:
        return i
    else:
        return i+1

def get_points_inside_boundary(b, points):
    pts = []
    for pt in points:
        ret = cv.pointPolygonTest(b, (pt[0],pt[1]), False)
        #print(pt,ret)
        if ret > 0:
            pts.append(pt)
    pts = np.array(pts)
    print("get %d point inside boudary"%(pts.shape[0]))
    return pts


def fill_points(pts, min_dis, max_dis=10):
    full = False
    print("point number before fill:%d"%(pts.shape[0]))
    while not full:
        full = True
        for i in range(pts.shape[0]):
            dis = distance(pts[i],pts[i-1])
            if dis>min_dis and dis<max_dis:
                #print(dis)
                full = False
                x = (pts[i][0]+pts[i-1][0])/2
                y = (pts[i][1]+pts[i-1][1])/2
                pts = np.insert(pts,i,[x,y],axis=0)
                
    print("point number after fill:%d"%(pts.shape[0]))    
    return pts



def select_points(pts, N=300):
    n = pts.shape[0]
    print("point number before select:%d"%(n))
    if n<N:
        print("sample number is too large!")
    step = n*1.0/N
    ret = []
    for i in range(N):
        ret.append(pts[math.floor(i*step)])
    print("point number selected:%d"%(N))
    ret = np.array(ret)
    return ret

def get_points_inside(boundary, resolution, N):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            pt = [j,i]
            points.append(pt)
    points = np.array(points)

    p_sparse = get_points_inside_boundary(boundary, points)
    p_dense = fill_points(p_sparse,min_dis=5, max_dis=10)
    p_dense = get_points_inside_boundary(boundary, p_dense)
    p_selected = select_points(p_dense, N)
    return p_selected

def boundary_match(bd1, bd2, k):
    b1 = bd1.copy()
    b2 = bd2.copy()

    N1 = b1.shape[0]
    N2 = b2.shape[0]

    x_m1, y_m1 = np.mean(b1,axis=0)
    x_m2, y_m2 = np.mean(b2,axis=0)
    print("mean 1: ",(x_m1,y_m1)," mean 2: ",(x_m2,y_m2))

    b1 += [int(x_m2-x_m1),int(y_m2-y_m1)]

    
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
            if j>prev[1] :#and j-prev[1]<=k):
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

def transfer(src_coo, src_img, tgt_co, background=0):
    src_co = src_coo.copy()
    tgt_img = np.zeros(src_img.shape)
    if background != 0:
        tgt_img = background - tgt_img
    #tgt_img = 255 - tgt_img
    R = src_img.shape[0]
    N = src_co.shape[0]
    for i in range(N):
        src = src_co[i]
        src = [to_closest(src[0])%R,to_closest(src[1])%R]
        tgt = tgt_co[i]   
        #print(tgt, src)     
        tgt_img[tgt[1]][tgt[0]] = src_img[src[1]][src[0]]
    return tgt_img

def test_img(img, coo, name):
    co = coo.copy()
    out = np.zeros(img.shape)
    N = img.shape[0]
    #out = 255 - out
    for p in co:
        p = [to_closest(p[0]),to_closest(p[1])]
        p = [p[0]%N, p[1]%N]
        out[p[1]][p[0]] = img[p[1]][p[0]]
    cv.imwrite(name,out)
 
def fill_hole(img, pts, it_num):
    N = img.shape[0]
    print("img N: ",N)
    channel = 3 #RGB
    if len(img.shape)<3:
        channel = 1 #gray
    white = [255]*channel
    black = [0]*channel
    for it in range(it_num):
        print("it ",it)
        for pt in pts:
            x,y = pt
            #print(holed_img[y][x])
            if y<N and x < N and (img[y][x]-white).any() and (img[y][x]-black).any() :
                continue
            p = np.zeros(channel)
            cnt = 0
            for i in range(y-1,y+2):
                for j in range(x-1,x+2):
                    if not (i<N and j < N ):
                        continue
                    if not (img[i][j]-white).all() or not(img[i][j]-black).all():
                        continue
                    cnt += 1
                    p += img[i][j]
            if cnt == 0:
                continue
            img[y][x] = p/cnt
    return img
