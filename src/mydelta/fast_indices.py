import numpy as np
import itertools
from .Tree import NodeType

def piter(iterable,n=1):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    for i in range(n):
        next(b, None)
    return zip(a, b)

def assign_index(clt):
    for ind,v in enumerate(clt.values()):
        v.ind = ind
    for v in clt.values():
        if v.node_type != NodeType.root:
            v.parent_ind = v.parent.ind
        if v.node_type != NodeType.terminal:
            v.children_ind = [v.child_L.ind,v.child_R.ind]
def assign_terminal_indices(clt,dd):
    ind = []
    child_ind = [] 
    for ii,v in enumerate(clt.values()):
        if v.node_type == NodeType.terminal:
            ind.append(dd[v.cell_class])
        else:
            lname = len(v.name) if v.name != 'root' else 0
            child_ind.append((ii,v.children_ind[0],v.children_ind[1],v.child_L.leaves,v.child_R.leaves,lname))
    return np.asarray(ind), np.row_stack(child_ind)
def grid_internal_terminal(internal:np.array,
                           len_terminal:int,
                           data_in_rows:bool,
                           mat_shape:tuple):
    m1 = np.lib.stride_tricks.as_strided(internal,
                                         np.append(len_terminal,internal.shape),
                                         np.append(0,internal.strides),
                                         writeable=False
                                         ).reshape((-1,internal.shape[-1]))
    i01 = np.arange(len_terminal)
    i1 = np.lib.stride_tricks.as_strided(i01,
                                         (len_terminal,internal.shape[0],3),
                                         (8,0,0),
                                         writeable=False
                                         ).reshape((-1,3))
    data = (m1[:,:3],i1) if data_in_rows else (i1,m1[:,:3])
    indices = np.ravel_multi_index(data,mat_shape if len(mat_shape) == 2 else mat_shape[1:])
    costs = m1[:,[4,3]] # 4 is for right child and 3 is for left child
    depth = m1[:,-1]
    if len(mat_shape) == 2:
        return indices, costs, depth
def grid_internal_internal(rows:np.array,
                           cols:np.array,
                           mat_shape:tuple):
    m_rows = np.lib.stride_tricks.as_strided(rows,
                                             np.append(len(cols),rows.shape),
                                             np.append(0,rows.strides),
                                             writeable=False
                                             ).reshape((-1,rows.shape[-1]))
    m_cols = np.lib.stride_tricks.as_strided(cols,
                                             np.append(len(rows),cols.shape),
                                             np.append(0,cols.strides),
                                             writeable=False
                                             ).reshape((-1,cols.shape[-1]))
    rind = m_rows[:,:3][:,np.repeat(np.arange(3),3)]
    cind = m_cols[:,:3][:,np.tile(np.arange(3),3)]
    indices = np.ravel_multi_index((rind,cind),mat_shape if len(mat_shape) == 2 else mat_shape[1:])
    # order is (a,b),(a,b_l),(a,b_r),(a_l,b),(a_l,b_l),(a_l,b_r),(a_r,b),(a_r,b_l),(a_r,b_r)
    # reorder indices
    indices = indices[:,[0,4,8,5,7,1,2,3,6]]
    # now order is (a,b),(a_l,b_l),(a_r,b_r),(a_l,b_r),(a_r,b_l),(a,b_l),(a,b_r),(a_l,b),(a_r,b)
    # adjust costs to correspond to the last four scenarios
    costs = np.column_stack((m_cols[:,4],m_cols[:,3],m_rows[:,4],m_rows[:,3]))
    depth = np.column_stack((m_rows[:,-1],m_cols[:,-1]))
    depth.sort(axis=1)
    if len(mat_shape)== 2:
        return indices, costs, depth
def adjust_indices_3d(indices,costs,depth,mat_shape):
    num_0 = mat_shape[0]
    arr_0 = np.arange(num_0)
    mul = mat_shape[1]*mat_shape[2]
    arr_0 *= mul
    i2 = (np.expand_dims(indices,0) + np.expand_dims(arr_0,[1,2])).reshape((-1,indices.shape[-1]))
    z = np.zeros(num_0)
    costs2 = (np.expand_dims(costs,0) + np.expand_dims(z,[1,2])).reshape((-1,costs.shape[-1]))
    depth2 = np.expand_dims(depth,0)
    arr2 = np.expand_dims(z,[1]) if depth2.ndim == 2 else np.expand_dims(z,[1,2])
    depth3 = (depth2 + arr2)
    depth3 = depth3.reshape((depth3.shape[0]*depth3.shape[1],-1)).squeeze()
    return i2, costs2, depth3    
def sort_indices_terminal(indices,costs,depth):
    asort_depth = np.argsort(depth)[::-1]
    sdepth = depth[asort_depth]
    #
    indices = indices[asort_depth]
    costs = costs[asort_depth]
    breaks = np.where(np.diff(sdepth) != 0)[0]+1
    breaks = np.append(0,np.append(breaks,len(sdepth)))
    breaks = [slice(*v) for v in piter(breaks)]
    return indices, costs, sdepth, breaks
def sort_indices_internal(indices,costs,depth):
    #asort_depth = np.argsort(depth)[::-1]
    asort_depth = np.lexsort(depth.T)[::-1]
    sdepth = depth[asort_depth]
    #
    indices = indices[asort_depth]
    costs = costs[asort_depth]
    #breaks = np.where(np.diff(sdepth) != 0)[0]+1
    breaks = np.where(~(np.diff(sdepth,axis=0)==0).all(1))[0]+1
    breaks = np.append(0,np.append(breaks,len(sdepth)))
    breaks = [slice(*v) for v in piter(breaks)]
    return indices, costs, sdepth, breaks
def get_all_indices(clt1,clt2,cost,num_rand):
    assign_index(clt1)
    assign_index(clt2)
    row_dict = dict(zip(cost.index.values,np.arange(cost.shape[0])))      
    col_dict = dict(zip(cost.columns.values,np.arange(cost.shape[1])))
    row_ind,row_children = assign_terminal_indices(clt1,row_dict)
    col_ind,col_children = assign_terminal_indices(clt2,col_dict)
    mat_shape = (len(clt1),len(clt2))
    dims = (num_rand+1,)+mat_shape
    row_indices,row_costs,row_depth = grid_internal_terminal(row_children,len(col_ind),True,mat_shape)
    col_indices,col_costs,col_depth = grid_internal_terminal(col_children,len(row_ind),False,mat_shape)
    t_indices = np.row_stack((row_indices,col_indices))
    t_costs = np.row_stack((row_costs,col_costs))
    t_depth = np.concatenate((row_depth,col_depth))
    t_indices,t_costs,t_depth = adjust_indices_3d(t_indices,t_costs,t_depth,dims)
    t_indices, t_costs, t_sdepth, t_breaks = sort_indices_terminal(t_indices,t_costs,t_depth)
    i_indices, i_costs, i_depth = grid_internal_internal(row_children,col_children,mat_shape)
    i_indices,i_costs,i_depth = adjust_indices_3d(i_indices,i_costs,i_depth,dims)
    i_indices, i_costs, i_sdepth, i_breaks = sort_indices_internal(i_indices, i_costs, i_depth)
    return locals()
