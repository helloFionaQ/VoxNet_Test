import copy

from voxnet import npytar as npyutil
import numpy as np
import config
import drawvox

'''
return:  the coordinate data of the rings in each not-null level
         the number of not-null levels
size  :  (levels(not-null), ring_size, 3 (dimensions) )

attention : the valid voxel's value is set to -1
'''



'''
get the z-th plane
input: a- 3D-array
       z- index of z
output: 2D plane of array

'''
V_VALID=-1

def get_plane(a,z):
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i][j][z]==1:
                array_zero = np.zeros([len(a),len(a)], dtype=int)
                array_zero[:, :,] = a[:, :,z]
                return array_zero
    return []

'''
get inside and outside edge
input: array - 2D plane array
       z - index of z axis
       
output: a - the edge voxel plane (2D)
        edge_list - the list of edge voxel coordinate
        
'''
def get_edge(array,z):
    a=np.zeros((dim,dim),dtype=int)
    # a=a-1
    flag=0
    edge_list=[]
    edge_out_list=[]
    start=1
    # temp=np.zeros((dim,dim),dtype=int)
    for i in range(dim):
        start=1
        for j in range(dim-1):
            if array[i][j]==1 :
                if start==1:
                    edge_out_list.append((i, j, z))
                    start=0
                if flag == 0:
                    edge_list.append((i, j, z))
                    a[i][j] = 1
                if array[i][j+1] == 1:
                    flag=1
                else:flag=0

    flag=0
    for j in range(dim):
        start=1
        for i in range(dim-1,0,-1):
            if array[i][j]==1:
                if start==1:
                    edge_out_list.append((i, j, z))
                    start=0
                if flag == 0:
                    a[i][j] = 1
                    edge_list.append((i, j, z))
                if array[i-1][j]==1:
                    flag=1
                else:flag=0

    flag=0
    for i in range(dim):
        start=1
        for j in range(dim-1,0,-1):

            if array[i][j]==1 :
                if start == 1:
                    edge_out_list.append((i, j, z))
                    start = 0
                if flag == 0:
                    a[i][j] = 1
                    edge_list.append((i, j, z))
                if array[i][j-1]==1:
                    flag=1
                else:flag=0

    flag=0
    for j in range(dim):
        start = 1
        for i in range(dim-1):

            if array[i][j]==1 :
                if start == 1:
                    edge_out_list.append((i, j, z))
                    start = 0
                if flag == 0:
                    a[i][j] = 1
                    edge_list.append((i, j, z))
                if array[i+1][j]==1:
                    flag=1
                else:flag=0
    edge_out_list=list(set(edge_out_list))
    edge_list=list(set(edge_list))

    return a,edge_list,edge_out_list

'''
get 2 vectors' distance
output: distance float
'''
def calcu_distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

'''
calculate the cos value of 2 vectors
'''
def calcu_cos(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)

'''
get center point
input : l - the coordinate list of plane array (2D)
        z - z axis of model
output : p - the coordinate of the center point
'''
def get_centerpoint_axis(l):
    lx=[]
    ly=[]
    for cor in l:
        lx.append(cor[0])
        ly.append(cor[1])
    return ((max(lx)+min(lx))//2,(max(ly)+min(ly))//2,l[0][2])


def get_centerpoint_elem(l):
    l_dis1 = []
    l_dis2 = []
    cor1 = np.array(l[0])
    for cor in l:
        l_dis1.append(calcu_distance(cor, cor1))
    l_dis1 = np.array(l_dis1)
    cor2 = np.array(l[l_dis1.argmax()])
    cor3 = np.array(l[l_dis1.argmax() // 2])

    for cor in l:
        l_dis2.append(calcu_distance(cor, cor3))
    l_dis2 = np.array(l_dis2)
    cor4 = np.array(l[l_dis2.argmax()])
    p = (cor1 + cor2 + cor3 + cor4) / 4
    return p

'''
get the vector made up with 2 points
input : the coordinates of the 2 points 
        b - the start point
output : the result vector
'''
def get_vector(a,b):
    v=((a[0]-b[0]),(a[1]-b[1]),(a[2]-b[2]))
    return v


fname=r'/home/haha/Documents/PythonPrograms/VoxNet_Test/voxnet-master/scripts/shapenet10_train.tar'
reader=npyutil.NpyTarReader(fname)
dim=config.cfg['dims'][0]
ring_size=90

'''
main function
input : the chunk voxel data arrays ( for example 2048 * 1 * 32*32*32 chunk_size, channel, dimension) 
output : arr_cors - the ring of each level(not null)  in coordinate
                    format : (levels, coordinates)
         arr_lsize - the level length of each model in the chunk
'''
def load_data(array):
    ls_cors = []
    ls_lsize=[]
    chunk_size=array.shape(0)
    for index in range(chunk_size):
        level_size=0
        x=list(array[index,0,:])
        # for each model
        for z in range(dim):
            # for each level
            ar_p=get_plane(x,z)
            if ar_p==[]:
                continue
            level_size+=level_size
            ar_e,ls_e,ls_eo=get_edge(ar_p,z)
            # get center point
            cor_center=get_centerpoint_axis(ls_eo)
            # get 2 parts of the ouside edge list (1:up 2:down)
            ls_eo_p1=[]
            ls_eo_p2=[]
            for cor in ls_eo:
                # cor= list(cor)   # change coordinate tuple into list
                if cor[0]<=cor_center[0]:
                    ls_eo_p1.append(cor)
                else:
                    ls_eo_p2.append(cor)
            # sorted the coordinate sequence
            ls_cos=[]
            ls_cor=[]
            v1=(0,-1,z)     # v1 : the start vector
            for cor in ls_eo_p1:
                cos=calcu_cos(get_vector(cor,cor_center),v1)
                ls_cos.append((cor,cos))
            ls_sort=sorted(ls_cos,key=lambda x:x[1],reverse=True)
            for i in ls_sort:
                ls_cor.append(i[0])
            ls_cos=[]
            for cor in ls_eo_p2:
                cos=calcu_cos(get_vector(cor,cor_center),v1)
                ls_cos.append((cor,cos))
            ls_sort=sorted(ls_cos,key=lambda x:x[1])
            for i in ls_sort:
                ls_cor.append(i[0])
            # set looping
            l=len(ls_cor)
            if l < ring_size:
                for i in range(l,ring_size):
                    ls_cor.append(ls_cor[i-l])
            ls_cors.append(ls_cor)
        ls_lsize.append(level_size)
    arr_lsize=np.array(ls_lsize)
    arr_cors=np.array(ls_cors)
    return arr_cors,arr_lsize

