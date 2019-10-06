import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time

from mv_cord.helper import *
from mv_cord.mv_cord import *
from mv_cord.visualization import *

from vertex_process.vertex import *
from img_process.process import *

# img resolution
resolution  = 1024

def get_corres(pts_from, bound_from, bound_to, file_name=None):
    mv_cord = get_mv_coordinate(pts_from, bound_from)
    if file_name != None:
        np.save(file_name,mv_cord)
    pts_to = np.dot(mv_cord, bound_to)
    return pts_to

def warp_img(src_pts, src_img, tgt_pts, file_name, background=0):
    trans = transfer(src_pts, src_img, tgt_pts,background)
    cv.imwrite(file_name,trans)
    return trans


def fill_inside_hole(holed_img, silh_img, it_num, file_name):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            pt = [j,i]
            points.append(pt)
    points = np.array(points)

    bound = get_boundary(silh_img)
   
    pts = get_points_inside_boundary(bound, points)#get all points insize boundary
    holed_img = fill_hole(holed_img, pts, it_num)
    holed_img = cv.GaussianBlur(holed_img,(5,5),0)
    cv.imwrite(file_name, holed_img) 
    return holed_img
  

def warp_xy(from_bd, from_pts, to_bd):    
    from_N = from_bd.shape[0]
    to_N = to_bd.shape[0]
    sp_N = min(from_N,to_N)

    # SMPL
    b1 = select_points(fill_points(from_bd,min_dis=3, max_dis=50),N=int(sp_N*0.8)) 
    # target
    b2 = select_points(fill_points(to_bd,min_dis=3, max_dis=50),N=int(sp_N*0.5))
    
    #draw_points(b1, r=resolution)
    #draw_points(b2, r=resolution)
    
    match = boundary_match(b1,b2,k=32)
    b1_m = b1[match]
    #draw_matching(b1_m, b2, r=resolution)

    # get correspond pts
    to_pts = get_corres(from_pts, b1_m, b2)
    return b1_m, b2, to_pts

def warp_depth(src_bd, to_bd, to_pts, sp_pts, img):
    to_pts = np.array(to_pts)
    sp_N = int(to_pts.shape[0] / 20)

    p_slc = select_points(fill_points(to_pts,min_dis=3, max_dis=50),N=sp_N) 
    src_pts = get_corres(p_slc, to_bd, src_bd)
    hole_im = transfer(src_pts, img, p_slc, 255)
    #cv.imwrite('tmp/holed_depth_%d.png'%idx,hole_im)
    filled_im = fill_hole(hole_im, to_pts, 1)
    filled_im = cv.blur(filled_im,(5,5))
    #cv.imwrite('tmp/filled_depth_%d.png'%idx,filled_im)
    return filled_im

def warp_z(init_dep, warp_dep, init_pts, warp_pts, bound, thresh=1):
    old_dep = get_img_value(init_dep, init_pts).astype(int)
    new_dep = get_img_value(warp_dep, warp_pts).astype(int)

    N = len(new_dep)
    bias = (new_dep-old_dep)/2500
    lim =  0.01
    
    for i in range(N):
        d = bias[i]
        if d > lim:
            d = lim
        if d < -lim:
            d = -lim
            
        dist = np.sqrt(np.sum(np.square(bound-warp_pts[i]),axis=1))
        if(np.min(dist)<thresh):
            d = 0
        bias[i] = d

    #bias = (new_dep-old_dep)/2550

    min_ = np.min(bias)
    max_ = np.max(bias)
    mm = np.mean(bias)
    #bias = (bias-min_-mm)*0.01/(max_-min_)
    
    print("bias: ",min_,max_,mm)

    return bias

def gen_mask_from_smpl(smpl_mask, rgb, dirpath):
    sample_num = 5000
    # SMPL
    b1 = get_boundary(smpl_mask,1,False)
    b1 = fill_points(b1,min_dis=3, max_dis=80)
    p1 = get_points_inside(b1, resolution, sample_num)
    b1 = select_points(b1,N=700) 
    #draw_points(p1, r=resolution)
    #draw_points(b1, r=resolution)

    # rgb
    b2 = get_boundary(rgb,1,False)
    b2 = fill_points(b2,min_dis=3, max_dis=80)
    #draw_points(b2, r=resolution)
    tp = get_points_inside(b2, resolution, sample_num)
 
    # match boundary
    b2 = select_points(b2,N=400)    
    match = boundary_match(b1,b2,32)
    b1_m = b1[match]
    #draw_matching(b1_m, b2, r=resolution)

    # get correspond pts
    sp = get_corres(tp, b2, b1_m)
    print("mean value...")
    hole = transfer(sp,smpl_mask,tp,0)  
    cv.imwrite(dirpath+"hole_mask.png",hole)
    print("fill hole...")
    fill = fill_mask(rgb,hole)
    print("img %d over"%idx)
    cv.imwrite(dirpath+"fill_mask.png",fill)
    
if __name__ == "__main__":
    #li = [100,700,900,1000,2400,2500,4300,4900]
    #li = [100,700,900,1000,2400,2500,4300,4900,7700,8200,8800,10000,10100]
    li = [0]
    for idx in li:
        #idx = 2400
        dr = 'file/%d/'%idx

        rgb = cv.imread(dr+'rgb.png')
        #rgb_mask = cv.imread(dr+'rgb_mask.png')
        #super_mask = refine_mask(rgb, rgb_mask.copy(), dr)


        # smpl_mask = cv.imread(dr+'smpl_mask.png')
        # smpl_depth = cv.imread(dr+'smpl_depth.png')
        # smpl_depth = cv.cvtColor(smpl_depth, cv.COLOR_BGR2GRAY)
        smpl_norm = cv.imread(dr+'smpl_normal.png')
        
        # gen_mask_from_smpl(smpl_mask,rgb,dr)

        smpl_seg_vts, smpl_seg_idxs = load_seg_vertex(dr+'%d.obj'%idx)
        smpl_seg_pts = []
        for i in range(Npart+1):
            vts = np.array(smpl_seg_vts[i])
            smpl_seg_pts.append(vert2pt(vts,resolution))
        
        # print('rgb mask')
        # rgb_mask_bouds,rgb_mask_pts = seg_mask(rgb_mask, rgb)
        # print('smpl mask')
        # smpl_mask_bouds,smpl_mask_pts = smpl_seg_mask(smpl_mask,smpl_seg_pts.copy())
        
        smpl_bond = get_boundary(smpl_norm,1,False)
        rgb_bond = get_boundary(rgb,1,False)

        x_m, y_m = np.mean(rgb_bond,axis=0)
        thresh = 30
        for pt in rgb_bond:
            if pt[1]<resolution/3:
                if pt[0]<x_m-30:
                    pt[0] += 3
                elif pt[0]>x_m+30:
                    pt[0] -= 3
            else:
                if pt[0]<x_m-55:
                    pt[0] += 6
                elif pt[0]>x_m+55:
                    pt[0] -= 6
                


        s_bd, r_bd, pts = warp_xy(smpl_bond, smpl_seg_pts[Npart].copy(), rgb_bond)

        # in all warp
        # s_bd, r_bd, pts = warp_xy(smpl_mask_bouds[Npart], smpl_seg_pts[Npart].copy(), rgb_mask_bouds[Npart])        
        # test_img(smpl_mask, smpl_seg_pts[Npart].copy(), dr+"tmp/all_smpl_xy.png")
        # test_img(rgb, pts, dr+"tmp/all_warp_xy.png")
        smpl_warp_vts_all = pt2vert(pts, np.zeros((6890,1)), smpl_seg_vts[Npart].copy(), resolution)

        np.save(dr+"warp_vts_all.npy",smpl_warp_vts_all)
        new_obj(dr+'%d.obj'%idx, dr+'out_all.obj', smpl_warp_vts_all, dr+'out.mtl')

        # # warp part by part
        # smpl_warp_vts_seg = np.zeros((6890,3))
        # smpl_warp_vts_seg_w = np.zeros((6890,3))
        # w_p = [0.85,0.8,0.8,0.8]
        # for i in range(Npart):
        #     # warp in x/y
        #     s_bd, r_bd, warpped = warp_xy(smpl_mask_bouds[i], smpl_seg_pts[i].copy(), rgb_mask_bouds[i])
        #     test_img(smpl_mask, smpl_seg_pts[i].copy(), dr+"tmp/%dsmpl_xy.png"%(i))
        #     test_img(rgb, warpped, dr+"tmp/%dwarp_xy.png"%(i))
            
        #     # warp in z
        #     #warp_dep = warp_depth(s_bd, r_bd, rgb_mask_pts[i], warpped.copy(), smpl_depth)
        #     #cv.imwrite(dr+'warp_depth%d.png'%i, warp_dep)
        #     #warp_dep = cv.imread(dr+'warp_depth%d.png'%i)
        #     #warp_dep = cv.cvtColor(warp_dep, cv.COLOR_BGR2GRAY)
        #     #z_bias = warp_z(smpl_depth, warp_dep, smpl_seg_pts[i].copy(), warpped.copy(), r_bd,10)

        #     z_bias = np.zeros((smpl_seg_pts[i].shape[0],1))
        #     warp_vts = pt2vert(warpped, z_bias, smpl_seg_vts[i].copy(), resolution)
        #     warp_vts_w = weight_avg_vts(warp_vts, smpl_seg_idxs[i], smpl_warp_vts_all, w_p[i])
        #     N = warp_vts.shape[0]
        #     for j in range(N):
        #         id_ = smpl_seg_idxs[i][j]
        #         smpl_warp_vts_seg[id_] = warp_vts[j]
        #         smpl_warp_vts_seg_w[id_] = warp_vts_w[j]
        
        # np.save(dr+"warp_vts_seg.npy",smpl_warp_vts_seg)
        # new_obj(dr+'%d.obj'%idx, dr+'out_seg.obj', smpl_warp_vts_seg, dr+'out.mtl')
        # new_obj(dr+'%d.obj'%idx, dr+'out_seg_wighted.obj', smpl_warp_vts_seg_w, dr+'out.mtl')

       
        
        # weighted avg warp
        # wp = weight_avg_vts(smpl_warp_vts_seg,smpl_warp_vts_all,smpl_seg_vts[Npart].copy())
        # new_obj(dr+'%d.obj'%idx, dr+'weight.obj', wp, dr+'out.mtl')
        

        print("-------------------------- model %d over ------------------------------ "%idx)
    
    
        
    

    
    


    
    
