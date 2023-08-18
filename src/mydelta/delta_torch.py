import torch
tc = torch.cuda
FloatTens = tc.FloatTensor
class TCLT(torch.nn.Module):
    def __init__(self,dims,cost_ij,row_ind,col_ind,t_indices,t_costs,
                 t_depth,t_breaks,i_indices,i_costs,i_depth,i_breaks,**kwargs):
        super().__init__()
        self.dims = tc.IntTensor(dims)
        self.cost = torch.nn.Parameter(FloatTens(cost_ij))
        self.pruning_cost = torch.nn.Parameter(FloatTens((1.0,)))
        self.row_ind = tc.LongTensor(row_ind)
        self.col_ind = tc.LongTensor(col_ind)
        self.t_indices = tc.LongTensor(t_indices)
        self.t_costs = FloatTens(t_costs)
        self.t_depth = tc.IntTensor(t_depth)
        self.t_breaks = t_breaks
        self.i_indices = tc.LongTensor(i_indices)
        self.i_costs = FloatTens(i_costs)
        self.i_depth = tc.IntTensor(i_depth)
        self.i_breaks = i_breaks
        self.ind = self.row_ind[:,None]*self.cost.shape[1]+self.col_ind[None,:]
        self.slr = slice(0,len(row_ind))
        self.slc = slice(0,len(col_ind))
        self.len_r = len(row_ind)
        self.len_c = len(col_ind)
        self.device = 'cuda'
    def forward(self):
        self.mat = FloatTens(size = tuple(self.dims))
        self.mat_flat = torch.flatten(self.mat)
        # use below r1, c1 for debugging
        #r1 = torch.row_stack([torch.arange(self.len_r,device=self.device) for ii in range(self.dims[0]-1)])
        #c1 = torch.row_stack([torch.arange(self.len_c,device=self.device) for ii in range(self.dims[0]-1)])
        r1 = torch.row_stack([torch.randperm(self.len_r,device=self.device) for ii in range(self.dims[0]-1)])
        c1 = torch.row_stack([torch.randperm(self.len_c,device=self.device) for ii in range(self.dims[0]-1)])
        ind2 = self.row_ind[r1][:,:,None]*self.cost.shape[1]+self.col_ind[c1][:,None,:]
        cost_flat = torch.nn.functional.softmax(self.cost.flatten(),dim=0)
        self.mat[0,self.slr,self.slc] = cost_flat[self.ind]
        self.mat[1:,self.slr,self.slc] = cost_flat[ind2]
        t_costs = self.t_costs * self.pruning_cost
        i_costs = self.i_costs * self.pruning_cost
        for sl in self.t_breaks:
            max_score = (self.mat_flat[self.t_indices[sl,1:]]-t_costs[sl]).max(1)
            self.mat.put_(self.t_indices[sl,0],max_score.values)
        for sl in self.i_breaks:
            scores = self.mat_flat[self.i_indices[sl,1:]]
            scores2 = torch.column_stack((scores[:,:2].sum(1),scores[:,2:4].sum(1),scores[:,4:] - i_costs[sl]))
            max_score = scores2.max(1)
            self.mat.put_(self.i_indices[sl,0],max_score.values)
        return self.mat[:,-1,-1]
    def loss(self):
        vals = self.forward()
        return (vals[0]-vals[1:].mean())/vals[1:].std()
    def zero_diag_grad(self):
        def hook(grad):
            grad = grad.clone()
            return grad.fill_diagonal_(0.0)
        self.hook = self.cost.register_hook(hook)
