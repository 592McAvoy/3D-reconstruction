import numpy as np
import cv2 as cv
from mv_cord.helper import to_closest

def load_obj_vertex(filename):
    
    f = open(filename, 'r')

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

    vertices = []
    
    for line in f:
        str = line.split()
        if len(str) == 0:
            continue

        if str[0] ==  '#' or str[0] == 'mtllib' or str[0] == 'usemtl':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:]]
            vertices.append( tmp_v )
        else:
            break

    f.close()

    v = np.asarray(vertices)

    return v

def vert2pt(vv, resolution):
    v = vv.copy()
    if len(v) == 0:
        return v;
    v = resolution * v
    for vt in v:
        vt[0] = resolution/2+vt[0]
        vt[1] = resolution - vt[1]
    return v[:,:2]  #(x,y)

def pt2vert(xy,z, old_vts, resolution):
    N = xy.shape[0]
    vts = np.zeros((N,3))
    old_vts = np.array(old_vts)
    z_mean = np.mean(old_vts,axis=0)[2]
    print("z_mean: ",z_mean)
    for i in range(N):
        x,y = xy[i]
        z_bia = z[i]
        vts[i][0] = (x - resolution/2)*1.0/resolution
        vts[i][1] = (resolution - y)*1.0/resolution
        vts[i][2] = old_vts[i][2]
        if old_vts[i][2]>z_mean/2:
            vts[i][2] += z_bia
    return vts

def dump_vert(vv,resolution, filename):
    v = vv.copy()
    for vt in v:
        vt[0] -= resolution/2
        vt[1] = resolution - vt[1]
    v /= resolution
    np.save(filename, v)

def get_img_value(img, vv):
    v = vv.copy()
    out = []
    N = img.shape[0]
    for p in v:
        p = [to_closest(p[0]),to_closest(p[1])]
        p = [p[0]%N, p[1]%N]
        vl = img[p[1]][p[0]]
        out.append(vl)
    out = np.asarray(out)
    return out
    
def dump_normal(norm_img, vv, filename):
    rgb = get_img_value(norm_img, vv)
    norm = (rgb/255.0 - [0.5,0.5,0.5])*2
    np.save(filename, norm)

def dump_depth(depth_old, vv_old, depth_new, vv_new, bound, thresh, filename):
    old_dep = get_img_value(depth_old, vv_old)
    new_dep = get_img_value(depth_new, vv_new)
    N = len(new_dep)
    for i in range(N):
        if new_dep[i] == 255:
            new_dep[i] = old_dep[i]

        dist = np.sqrt(np.sum(np.square(bound-vv_new[i]),axis=1))
        if np.min(dist) < thresh:
            #print("idx: ",i," dist min:", np.min(dist))
            new_dep[i] = old_dep[i]

    minus = new_dep-old_dep
    np.save(filename, minus)


def write_obj( mesh_v, mesh_f_front, mesh_f_back, mesh_vt, mesh_vn, filepath, mtl_file=None, texture_name=None, verbose=True):
    with open( filepath, 'w') as fp:
        if mtl_file != None:
            li = mtl_file.split('/')
            write_mtl(mtl_file, texture_name)
            fp.write('mtllib ' + li[-1] +'\n')
            

        for v in mesh_v:
            if len(v) == 3:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2] ) )
            elif len(v) == 6:
                fp.write( 'v %f %f %f %f %f %f\n' % ( v[0], v[1], v[2], v[3], v[4], v[5] ) )
        if mesh_vt is not None:
            for t in mesh_vt:
                fp.write( 'vt %f %f\n' % ( t[0], t[1] ) )
        if mesh_vn is not None:
            for n in mesh_vn:
                fp.write( 'vn %f %f %f\n' % ( n[0], n[1], n[2] ) )
        
        def write_f(mesh_f):
            for f in mesh_f+1: # Faces are 1-based, not 0-based in obj files
                if f.shape == (1,3) or f.shape == (3,) or f.shape == (3,1):
                    #fp.write( 'f %d %d %d\n' % ( f[0],f[1],f[2]) )
                    fp.write( 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % ( f[0],f[0],f[0],f[1],f[1],f[1],f[2],f[2],f[2]) )
                elif f.shape == (3,2):
                    fp.write( 'f %d/%d %d/%d %d/%d\n' % ( f[0,0],f[0,1],f[1,0],f[1,1],f[2,0],f[2,1]) )
                elif f.shape == (3,3):
                    fp.write( 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % ( f[0,0],f[0,1],f[0,2],f[1,0],f[1,1],f[1,2],f[2,0],f[2,1],f[2,2]) )
                else:
                    print("strange faces shape!")
        if mesh_f_front is not None:
            fp.write('usemtl FRONT\n')
            write_f(mesh_f_front)
        if mesh_f_back is not None:
            fp.write('usemtl BACK\n')
            write_f(mesh_f_back)
        

    if verbose:
        print ('mesh saved to: ', filepath) 

def load_obj(filename):
    
    f = open(filename, 'r')

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

    vertices = []
    faces = []
    texcoords = []
    normals = []

    for line in f:
        str = line.split()
        if len(str) == 0:
            continue

        if str[0] ==  '#':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:]]
            vertices.append( tmp_v )

        elif str[0] == 'f':
            tmp_f = [get_num(s.split("/"), 'list') for s in str[1:]]
            faces.append( tmp_f )

        elif str[0] == 'vt':
            tmp_vt = [get_num(s, 'float') for s in str[1:]]
            texcoords.append( tmp_vt )

        elif str[0] == 'vn':
            tmp_vn = [get_num(s, 'float') for s in str[1:]]
            normals.append( tmp_vn )

    f.close()

    v = np.asarray(vertices)
    vn = np.asarray(normals)
    f = np.asarray(faces)
    vt = np.asarray(texcoords)

    if len(vt)>0:
        vt = vt[:,:2]

    return ( v, f, vt, vn )

def append_f(f):
    A,B,C = f.shape
    ff = np.zeros((A,B,C))
    for i in range(A):
        line = f[i]
        for j in range(B):
            group = line[j]
            #group = np.append(group, group[0])
            group[-1]=group[0]
            ff[i][j] = group
    return ff

def divide_face(f,v):
    N = f.shape[0]  # N face
    z_vec = [0,0,1]

    front = []
    back = []
    for i in range(N):
        if i==0:
            print(f[i][0][0])
        v0 = v[f[i][0][0]]
        v1 = v[f[i][1][0]]
        v2 = v[f[i][2][0]]

        m = v1-v0
        n = v2-v1

        # normal
        x = m[1]*n[2]-n[1]*m[2]
        y = n[0]*m[2]-m[0]*n[2]
        z = m[0]*n[1]-n[0]*m[1]

        if z>=0 :
            front.append(f[i])
        else:
            back.append(f[i])
    
    front = np.array(front)
    back = np.array(back)

    return front,back


def create_new_obj(old_obj_name, new_obj_name, v_file, dep_file, mtl_name):
    v, f, vt, vn = load_obj(old_obj_name)

    N = v.shape[0]
    v_xy = np.load(v_file)  
    dep = np.load(dep_file) 
    
    min = np.min(dep)
    max = np.max(dep)
    mm = np.mean(dep)
    dep = (dep-min-mm)*0.1/(max-min)
    
    mean = np.mean(dep)
    print(mean)
    for i in range(N):
        p = v_xy[i]
        d = dep[i]
        v[i][0] = p[0]
        v[i][1] = p[1]
        if d>mean:
            d=mean
        if d<-mean:
            d=-mean
        v[i][2] += d
    
    ff = append_f(f)
    #norm = np.load(vn_file)

    write_obj(v,ff,v_xy+[0.5,0],vn, new_obj_name, mtl_name, 'rgb.png')
    #write_obj(v,f,vt,vn,new_obj_name)

def new_obj(old_obj_name, new_obj_name, vts, mtl_name):
    v, f, vt, vn = load_obj(old_obj_name)

    #f = append_f(f)
    ff,bf = divide_face(f,vts)
    #norm = np.load(vn_file)

    write_obj(vts, ff,bf, vts[:,:2]+[0.5,0], vn, new_obj_name, mtl_name, 'rgb.png')
    #write_obj(v,f,vt,vn,new_obj_name)

def write_mtl(filename, imgname):
    with open( filename, 'w') as fp:
        fp.write('newmtl Solid\nKa  1.0 1.0 1.0\nKd  1.0 1.0 1.0\nKs  0.0 0.0 0.0\nd  1.0\nNs  0.0\nillum 0\n')
        fp.write("\n")
        fp.write('newmtl FRONT\nKa  1.0 1.0 1.0\nKd  1.0 1.0 1.0\nKs  0.0 0.0 0.0\nd  1.0\nNs  0.0\nillum 0\n')
        fp.write('map_Kd '+imgname+'\n')
        fp.write("\n")
        fp.write('newmtl BACK\nKa  1.0 1.0 1.0\nKd  1.0 1.0 1.0\nKs  0.0 0.0 0.0\nd  1.0\nNs  0.0\nillum 0\n')
        fp.write('map_Kd back.png\n')
    
if __name__ == "__main__":
    v,f,vt,vn = load_obj('7000.obj')
    ff,bf = divide_face(f,v)
    print(ff.shape[0])
    print(bf.shape[0])
    #write_obj(v,f[:,:,0],None,None,'out.obj')
    write_obj(v,ff,bf,vt,vn, 'out.obj', 'out.mtl', 'rgb.png')

