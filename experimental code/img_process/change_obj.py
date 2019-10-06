import numpy as np
import os
from shutil import copyfile

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

def process_obj(mesh_v, scale):
    center = np.mean(mesh_v, axis=0)
    mesh_v -= center
    max_val = np.max(mesh_v)
    mesh_v /= max_val
    mesh_v *= scale
    return mesh_v


def write_obj( mesh_v, mesh_f, mesh_vt, mesh_vn, filepath, verbose=True):
    with open( filepath, 'w') as fp:
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
        for f in mesh_f+1: # Faces are 1-based, not 0-based in obj files
            if f.shape == (1,3) or f.shape == (3,) or f.shape == (3,1):
                fp.write( 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % ( f[0],f[0],f[0],f[1],f[1],f[1],f[2],f[2],f[2]) )
            elif f.shape == (3,2):
                fp.write( 'f %d/%d %d/%d %d/%d\n' % ( f[0,0],f[0,1],f[1,0],f[1,1],f[2,0],f[2,1]) )
            elif f.shape == (3,3):
                fp.write( 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % ( f[0,0],f[0,1],f[0,2],f[1,0],f[1,1],f[1,2],f[2,0],f[2,1],f[2,2]) )
            else:
                print("strange faces shape!")

    if verbose:
        print ('mesh saved to: ', filepath) 

    
if __name__ == "__main__":
    f = open("meta.txt",'r')
    cnt = 0
    for line in f:
        if cnt > 2000:
            break
        cnt = cnt + 1
        in_file = "./comb_obj/"+line[:-1]+".obj"
        out_file = "./scaled_obj/"+line[:-1]+".obj"
        v, f, vt, vn = load_obj(in_file)
        v_out = process_obj(v, 10)
        write_obj(v_out,f,vt,vn,out_file)
