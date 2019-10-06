import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

#Npart = 5   # head, body, lhand, rhand, leg
Npart = 4   # lhand, rhand, other, head

def draw_points(pts, r):
    pts=pts.reshape(pts.shape[0],2)
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




def get_boundary(img, thresh, combine=True):      
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, thresh, 255, 0)
    thresh = cv.medianBlur(thresh,5)

    # kernel = np.ones((7,7),np.uint8)  
    # thresh = cv.erode(thresh, kernel,iterations = 1)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    N = len(contours)
    contour = []
    print(N)
    if combine:
        for i in range(N):        
            c = contours[i]
            if(len(c)<100):
                continue
            if contour ==[]:
                contour = c
            else:
                contour = np.concatenate((c,contour),axis=0)
    else:
        contour = contours[0]
    contour = contour.reshape(contour.shape[0],2)
    print("boudary before fill - shape :",contour.shape)      
    return contour

def get_seg_color(img):
    black = [0]*3
    N = img.shape[0]
    dic = []
    mark = []
    for i in range(N):
        for j in range(N):
            if (not (img[i][j]-black).any()) and (img[i][j-1]-black).any() and (img[i][j+1]-black).any():
                mid = j
            if (img[i][j]-black).any():
               col = (img[i][j][0],img[i][j][1],img[i][j][2])
               if dic.count(col)==0:
                   dic.append(col)
                   mark.append(1)
               else:
                   mark[dic.index(col)]+=1
    for i in range(len(dic)):
       if mark[i]>100:
           print(dic[i],": ",mark[i])

#file hole in body
def fill_body_hole(img, pts):
    N = img.shape[0]
    left = []
    right = []

    # top_down search
    shoudler = 0
    s_pos = [0,0]  
    first = 0  
    end = N
    for i in range(N):
        line = img[i]
        dic = np.nonzero(line)[0]
        if len(dic) == 0 or max(dic) - min(dic)<10 :
            if first != 0 and i-first>200:
                end = i
                break
            continue;
        if first==0:
            first = i # mark 1st line is not black
        tmp = [min(dic), max(dic)]
        if left != [] and shoudler==0 and (tmp[1]-tmp[0]<right[-1]-left[-1]):
            shoudler = i
            s_pos = [left[-1],right[-1]]
        left.append(tmp[0])
        right.append(tmp[1])

    # fill holes
    ml = int(np.mean(left))
    mr = int(np.mean(right))
    mid = int(first+(end-first)/3)
    print("body avg (%d,%d) , body start:%d mid:%d end:%d "%(ml,mr, first, mid,end))
    
    N = len(left)
    for i in range(mid-100-first,mid+300-first):
        rl = left[i]
        rr = right[i]
        if rr-rl < mr-ml-30:
            pl = int(0.3*rl + 0.7*ml)
            pr = int(0.3*rr + 0.7*mr)
            l = min(rl,pl)
            r = max(rr,pr)
            img[i+first,l:r] = 255
            for k in range(l,r):
                p = (k,i+first)
                pts.append(p)            

    return img

#file hole in leg
def fill_leg_hole(leg_im, leg_pts,body_im, body_pts):
    N = leg_im.shape[0]
    left = []
    right = []
    first = 0

    # bottom-up search
    start = 0
    for i in range(N-1,0,-1):
        line = leg_im[i]
        dic = np.nonzero(line)[0]
        if len(dic) == 0 :
            if first != 0:
                start = i
                break
            continue;
        if first==0:
            first = i # mark 1st line is not black
        tmp = [min(dic), max(dic)]
        left.append(tmp[0])
        right.append(tmp[1])

    ml = int(np.mean(left))
    mr = int(np.mean(right))
    N = len(left)
    for i in range(0,55):
        b = int((50-i)/3)
        rl = left[N-i-1]
        rr = right[N-i-1]
        pl = int(0.5*rl + 0.5*ml)
        pr = int(0.5*rr + 0.5*mr)
        l = min(rl,pl)
        r = max(rr,pr)
        body_im[i+start,l:r] = 255
        #leg_im[i+start,l:r] = 255
        if i < 50:
            leg_im[i+start,:] = 0
        for k in range(l,r):
            p = (k,i+start)
            if i<50 and leg_pts.count(p) > 0 :
                leg_pts.remove(p)
                #leg_pts.append(p)
            body_pts.append(p)
            

    return leg_im, body_im

#file hole in neck
def fill_neck_hole(head_im, body_im, body_pts):
    N = head_im.shape[0]
    left = []
    right = []
    first = 0

    # top-down search
    down = 0
    for i in range(N):
        line = head_im[i]
        dic = np.nonzero(line)[0]
        if len(dic) == 0 :
            if first != 0:
                down = i
                break
            continue;
        if first==0:
            first = i # mark 1st line is not black
        tmp = [min(dic), max(dic)]
        left.append(tmp[0])
        right.append(tmp[1])

    N = len(left)
    for i in range(0,18):
        l = left[N-i-1]
        r = right[N-i-1]
        body_im[down-i-1,l:r] = 255
        for k in range(l,r):
            p = (k,down-i-1)
            body_pts.append(p)        

    return body_im

def seg_mask(img, rgb_img, super_mask=None):
    
    black = [0]*3
    HEAD = (128,0,255)
    BODY = (0,0,128)
    HAND = (0,255,255)
    LEG = (0,255,0)
    N = img.shape[0]

    seg_pts = []
    seg_bouds = []
    hands = []
    imgs = np.zeros((Npart,N,N))
    for i in range(Npart+1):
        seg_pts.append([])
        seg_bouds.append([])
    mid = N/2
    cur = HEAD
    color_rec = [HEAD, BODY, HAND, LEG]
    for i in range(N):
        for j in range(N):
            pt = (j,i)
            # rgb中为黑色
            if not (rgb_img[i][j]-black).any():
                continue
            else:
                # rgb中有颜色，mask中是黑色 -> hole               
                if not (img[i][j]-black).any():
                    # super mask中有颜色，就用这个颜色
                    if super_mask is not None:
                        if (super_mask[i][j]-black).any():
                            img[i][j] = super_mask[i][j]
                    else:
                        li = []
                        cnt = []
                        for m in range(i-2,i+3):
                            for n in range(j-2,j+3):
                                if m<N and n<N and (img[m][n]-black).any():
                                    c = (img[m][n][0],img[m][n][1],img[m][n][2])
                                    if li.count(c)==0:
                                        li.append(c)
                                        cnt.append(1)
                                    else:
                                        cnt[li.index(c)]+=1
                        if len(cnt)>0:
                            img[i][j] = li[cnt.index(max(cnt))]                            
                                
                # 根据颜色划分到不同的区域
                col = (img[i][j][0],img[i][j][1],img[i][j][2])                

                if col != cur and i>N*3/4:
                    col = LEG
                cur = col

                if super_mask is not None:
                    col_s = (super_mask[i][j][0],super_mask[i][j][1],super_mask[i][j][2])
                    if col_s == HAND and i<N*2/3:
                        hands.append([j,i])

                # if col == HEAD:
                #     seg_pts[0].append(pt)        # HEAD
                #     imgs[0][i][j]=255

                # elif col == BODY:
                #     seg_pts[1].append(pt)        # BODY
                #     imgs[1][i][j]=255

                # elif col == HAND:
                #     hands.append([j,i])

                # elif col == LEG: 
                #     seg_pts[4].append(pt)        # LEG
                #     imgs[4][i][j]=255

                
                if col == HAND:
                    hands.append([j,i])
                elif col == HEAD:
                    seg_pts[3].append(pt)        # HEAD
                    imgs[3][i][j]=255
                else:
                    seg_pts[2].append(pt)        # LEG
                    imgs[2][i][j]=255
                   

    # divide hands into left and right
    mx,my = np.mean(np.array(hands),axis=0)
    for pt in hands:
        x,y = pt
        if x < mx:
            seg_pts[0].append((x,y))    # LEFT HAND
            imgs[0][y][x]=255

        else:
            seg_pts[1].append((x,y))    # RIGHT HAND
            imgs[1][y][x]=255

    for i in range(Npart):       
        im = imgs[i]
        if i == 2:
            im = fill_body_hole(im, seg_pts[2])
        #     im = fill_neck_hole(imgs[0], im, seg_pts[i])
        # if i == 4:
        #     im, imgs[1] = fill_leg_hole(im, seg_pts[i], imgs[1], seg_pts[1])
        cv.imwrite('tmp/%d.png'%i,im)
        im = cv.imread('tmp/%d.png'%i)
        seg_bouds[i] = get_boundary(im,128)

    seg_bouds[Npart] = get_boundary(rgb_img,1,False)
   
    return seg_bouds,seg_pts

def fill_im(img, skip_min=0, skip_max=1000):
    N = img.shape[0]    
    for i in range(N):        
        line = img[i]
        dic = np.nonzero(line)[0]
        if len(dic) == 0 :
            continue
        if i<skip_min :
            continue
        if i>skip_max:
            break;
        img[i,min(dic):max(dic)] = 255
    return img

def fill_smpl_leg_hole(img):
    N = img.shape[0]    

    for i in range(N):        
        line = img[i]
        dic = np.nonzero(line)[0]
        if len(dic) == 0 or min(dic)>512 or max(dic)<512 or i<400 :
            continue
        print(min(dic),max(dic))
        img[i:i+35,min(dic):max(dic)] = 255
        break
    return img
   

def smpl_seg_mask(img, smpl_pts):
    black = [0]*3

    # BGR
    HEAD = (255,0,255)
    BODY = (0,0,255)
    LHAND = (255,255,0)
    RHAND = (0,255,255)
    LLEG = (0,255,0)
    RLEG = (255,0,0)

    N = img.shape[0]
    seg_pts = []
    seg_bouds = []
    imgs = np.zeros((Npart,N,N))
    for i in range(Npart+1):
        seg_pts.append([])
        seg_bouds.append([])
    mid = N/2
    for i in range(N):
        for j in range(N):
            pt = (j,i)
            if (img[i][j]-black).any():
                col = (img[i][j][0],img[i][j][1],img[i][j][2])
                # if col == HEAD:
                #     seg_pts[0].append(pt)        # HEAD
                #     imgs[0][i][j]=255
                # elif col == BODY:
                #     seg_pts[1].append(pt)        # BODY
                #     imgs[1][i][j]=255
                # elif col == LHAND:
                #     seg_pts[2].append(pt)    # LEFT HAND
                #     imgs[2][i][j]=255
                # elif col == RHAND:  
                #     seg_pts[3].append(pt)    # RIGHT HAND
                #     imgs[3][i][j]=255
                # elif col == LLEG or col == RLEG:  
                #     seg_pts[4].append(pt)    # LEG
                #     imgs[4][i][j]=255
                
                if col == LHAND:
                    seg_pts[0].append(pt)    # LEFT HAND
                    imgs[0][i][j]=255
                elif col == RHAND:  
                    seg_pts[1].append(pt)    # RIGHT HAND
                    imgs[1][i][j]=255
                elif col == HEAD:
                    seg_pts[3].append(pt)        # HEAD
                    imgs[3][i][j]=255
                else: 
                    seg_pts[2].append(pt)    # LEG
                    imgs[2][i][j]=255
                    

    skip=[0,0,30,30,0] # skip neck and shoudler
    for i in range(Npart):     
        pts = smpl_pts[i]  
        im = imgs[i]
        for pt in pts:
            x,y = pt
            im[int(y)%N][int(x)%N] = 255
        if i == Npart-2:
            im = fill_im(im,300, 680)
        # else:
        #     im = fill_im(im)
        # if i < Npart-1:
        #     im = fill_im(im, skip[i])
        # else:
        #     im = fill_smpl_leg_hole(im)
        cv.imwrite('tmp/smpl%d.png'%i,im)
        im = cv.imread('tmp/smpl%d.png'%i)
        seg_bouds[i] = get_boundary(im,128)
    
    seg_bouds[Npart] = get_boundary(img,1,False)
    return seg_bouds,seg_pts

def get_num(string, type):
        if type == 'int':
            return int(string)
        elif type == 'float':
            return float(string)
        elif type == 'list':
            li = []
            for s in string:
                if s is not "":
                    li.append(int(s)-1)
                else:
                    li.append(0)
            return li
        else:
            print( 'Wrong type specified')

def load_seg_vertex(filename):
    f = open(filename, 'r')
    
    vertices = []
    idxs = []
    for i in range(Npart+1):
        vertices.append([])
        idxs.append([])

    # RGB
    HEAD = (255,0,128)
    BODY = (128,0,0)
    LHAND = (0,192,192)
    RHAND = (255,255,0)
    LLEG = (0,0,128)
    RLEG = (0,128,0)

    idx = 0
    for line in f:
        str = line.split()
        if len(str) == 0:
            continue
        if str[0] ==  '#' or str[0] == 'mtllib' or str[0] == 'usemtl':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:4]]
            vertices[Npart].append(tmp_v) #record all vts

            # tmp_c = [get_num(s, 'float') for s in str[4:]]
            # col = (int(tmp_c[0]),int(tmp_c[1]),int(tmp_c[2]))           
            
            # if col == LHAND:
            #     vertices[0].append(tmp_v)
            #     idxs[0].append(idx)
            # elif col == RHAND:
            #     vertices[1].append(tmp_v)
            #     idxs[1].append(idx)
            # elif col == HEAD:
            #     vertices[3].append(tmp_v)
            #     idxs[3].append(idx)
            # else:
            #     vertices[2].append(tmp_v)
            #     idxs[2].append(idx)  
                
            # idx += 1
                
        else:
            break

    f.close()

    return vertices,idxs

def refine_mask(rgb, mask, save_path):
    im = rgb.copy()
    im = cv.blur(im,(5,5))
    out = np.zeros_like(mask)
    # calculate super pixel
    slic = cv.ximgproc.createSuperpixelSLIC(im, algorithm=cv.ximgproc.MSLIC,region_size=30, ruler=30.0)
    slic.iterate(2)
    lb = slic.getLabelContourMask()
    cv.imwrite(save_path+"label.png",lb)
    labels = slic.getLabels()
    n = slic.getNumberOfSuperpixels()
    print("label n: ",n)
    black = [0]*3
    for i in range(n):
        area = np.where(labels.copy()==i)
        l = len(area[0])
        pts = np.concatenate((area[0].reshape(-1,1),area[1].reshape(-1,1)),axis=1)
        rec = []
        cnt = []
        for j in range(l):
            y,x = pts[j]
            c = mask[y][x]
            col = (c[0],c[1],c[2])
            if rec.count(col) == 0:
                rec.append(col)
                cnt.append(1)
            else:
                cnt[rec.index(col)] += 1
        #print(cnt)
        ch = rec[cnt.index(max(cnt))]
        #print("area %d :len %d - "%(i,l), ch)
        choice = np.array([ch[0],ch[1],ch[2]])
        if (choice- black).any():
            for k in range(l):
                out[pts[k][0]][pts[k][1]] = choice
        #cv.imwrite("area_%d.png"%i,out)
    cv.imwrite(save_path+"refine_mask.png",out)
    print("mask refined")
    return out


def fill_mask(rgb, hole_mask):             
    N = rgb.shape[0]      
    img = hole_mask.copy()
    black = [0]*3
    for i in range(N):
        for j in range(N):
        # rgb中为黑色
            if not (rgb[i][j]-black).any():
                continue
            else:
                # rgb中有颜色，hole_mask中是黑色 -> hole               
                if not (img[i][j]-black).any():               
                    li = []
                    cnt = []
                    for m in range(i-4,i+5):
                        for n in range(j-4,j+5):
                            if m<N and n<N and (hole_mask[m][n]-black).any():
                                c = (hole_mask[m][n][0],hole_mask[m][n][1],hole_mask[m][n][2])
                                if li.count(c)==0:
                                    li.append(c)
                                    cnt.append(1)
                                else:
                                    cnt[li.index(c)]+=1
                    if len(cnt)>0:
                        img[i][j] = li[cnt.index(max(cnt))]
    return img                             

def weight_avg_vts(part, part_ids, whole, w_p=0.7):
    out = np.zeros_like(part)
    N = part.shape[0]

    def dist(p1,p2):
        return np.sqrt(np.sum(np.square(p1-p2),axis=1))

    for i in range(N):
        idx = part_ids[i]
        part_p = part[i]
        whole_p = whole[idx]
        if part_p[1] < whole_p[1]-0.01:
            out[i] = 0.95*whole_p + 0.05*part_p
        else:
            out[i] = (1-w_p)*whole_p + w_p*part_p
    
    return out


if __name__ == "__main__":
    rgb = cv.imread('rgb.png')
    mask = cv.imread('hole_mask.png')
    f_mask = fill_mask(rgb,mask)
    cv.imwrite("fill_mask.png",f_mask)

    # slic = cv.ximgproc.createSuperpixelSLIC(rgb, algorithm=cv.ximgproc.MSLIC,region_size=50, ruler=30.0)
    # print(type(slic))
    # slic.iterate(2)
   
    # n =	slic.getNumberOfSuperpixels()
    # print("labels num: ",n)
    # labels = slic.getLabels()
    # print(type(labels))
    # print(labels.shape)
    # print(labels[512,450:500])
    # im = slic.getLabelContourMask()
    # print(im.shape)
    # #cv.imshow("super",im)
    # cv.imwrite("super.png",im)

    
    

