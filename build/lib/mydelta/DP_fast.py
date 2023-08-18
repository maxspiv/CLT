from mydelta.fast_indices import get_all_indices

def run_fast(dims,row_ind,col_ind,cost2,prune_cost,
             t_indices,t_costs,t_breaks,
             i_indices,i_costs,i_breaks,
             with_rand=True):
    mat = zeros(dims)
    ind = row_ind[:,None]*cost2.shape[1]+col_ind[None,:]
    num_rand_range = range(dims[0]-1)
    if with_rand:
        r1 = row_stack([permutation(len(row_ind)) for ii in num_rand_range])
        c1 = row_stack([permutation(len(col_ind)) for ii in num_rand_range])
    else:
        r1 = row_stack([arange(len(row_ind)) for ii in num_rand_range])
        c1 = row_stack([arange(len(col_ind)) for ii in num_rand_range])
    slr = slice(0,len(row_ind))
    slc = slice(0,len(col_ind))
    ind2 = row_ind[r1][:,:,None]*cost2.shape[1]+col_ind[c1][:,None,:]
    mat[0,slr,slc] = cost2.values.flat[ind]
    mat[1:,slr,slc] = cost2.values.flat[ind2]
    #
    mat_flat = mat.flat
    for sl in t_breaks:
        max_score = (mat_flat[t_indices[sl,1:]]-t_costs[sl]*prune_cost).max(1)
        np.put(mat,t_indices[sl,0],max_score)
    #
    for sl in i_breaks:
        scores = mat_flat[i_indices[sl,1:]]
        scores2 = column_stack((scores[:,:2].sum(1),scores[:,2:4].sum(1),scores[:,4:] - i_costs[sl]*prune_cost))
        max_score = scores2.max(1)
        np.put(mat,i_indices[sl,0],max_score)
    return mat

if __name__ == '__main__':
    import inspect
    mat_size = 2
    zind = get_all_indices(clt1,clt2,cost2,mat_size-1)
    signature = inspect.signature(run_fast)
    args = {k:zind.get(k,None) for k in signature.parameters.keys()}
    args['with_rand'] = False
    args['cost2'] = zind['cost']
    args['prune_cost'] = 1.0
    mat = run_fast(**args)
    
