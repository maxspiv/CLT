import pandas as pd
from mydelta.Tree import nodes_from_file,create_all
from mydelta.DP_recursive import DynProg, Backtrack,BtType

import delta
def run_hsa(fname1,fname2,cost_file):
    hsa = delta.HSA()
    hsa.BuildCost(cost_file)
    hsa.BuildTreeTS(fname1,fname2)
    hsa.DP()
    hsa.outputScoreMatrix_path('/tmp/score.alm')
    hsa.outputBtMatrix('/tmp/btmat.alm')
    hsa.outputTree_path('/tmp/tree1',hsa.treeS)
    hsa.outputTree_path('/tmp/tree2',hsa.treeT)
    hsa = delta.HSA()
    hsa.BuildCost(cost_file)
    hsa.BuildTreeTS(fname1,fname2)
    hsa.outputGResult('/tmp/out.alm')
    treeS = pd.read_csv('/tmp/tree1',sep='\t')
    treeT = pd.read_csv('/tmp/tree2',sep='\t')
    rr = pd.read_csv('/tmp/score.alm',sep='\t',skiprows=2,skipfooter=1,index_col=False,header=None,engine='python').iloc[:,:-1]
    bt = pd.read_csv('/tmp/btmat.alm',sep='\t',skiprows=2,skipfooter=1,index_col=False,header=None,engine='python').iloc[:,:-1]
    #print(rr)
    ind = ['q'+v for v in treeS.id.values]
    cols = ['s'+v for v in treeT.id.values]
    ind[-1] = cols[-1] = 'root'
    rr.index = ind
    rr.columns = cols
    bt.index = rr.index
    bt.columns = rr.columns
    return rr,bt

bt_trans = {
    1:BtType.NONE,
    2:BtType.Prune_2_R,
    3:BtType.Prune_2_L,
    4:BtType.Prune_1_R,
    5:BtType.Prune_1_L,
    6:BtType.Match_L_L,
    7:BtType.Match_L_R}
def translate_bt(btmat):
    btm2 = array([bt_trans[v].name for v in btmat.values.flat]).reshape(btmat.shape)
    btm3 = pd.DataFrame(btm2,btmat.index,btmat.columns)
    return btm3

def tree_file(t1,fname):
    df = pd.DataFrame(t1,columns=['Lineage','Name','Class'])
    fname2 = f'/tmp/{fname}.alm'
    df_csv = df.to_csv(None,sep='\t',index=False)
    open(fname2, 'w').write(df_csv[:-1])
    
if __name__ == '__main__':
    t1 = [['00','A','A'],['01','B','B'],['1','C','C']]
    t2 = [['000','B','B'],['001','C','C'],['01','A','A'],['1','C','C']]
    scores = pd.DataFrame([[2,0,-2],[0,2,-1],[-2,-1,2]],index=['A','B','C'],columns=['A','B','C']).stack()
    scores.to_csv('/tmp/cost.tsv',sep='\t',header=False) 
    
    tree_file(t1,'t1')
    tree_file(t2,'t2')
    
    fname1 = '/tmp/t1.alm'
    fname2 = '/tmp/t2.alm'
    cost_file = '/tmp/cost.tsv'

    fname1 = '/home/fix/work/max/DELTA/data/fun.alm'
    fname2 = '/home/fix/work/max/DELTA/data/pma.alm'
    cost_file = '/home/fix/work/max/DELTA/data/cost.tsv'

    rr,bt = run_hsa(fname1,fname2,cost_file)
    bt_orig = translate_bt(bt)
    nodes1 = nodes_from_file(fname1)
    nodes2 = nodes_from_file(fname2)
    clt1,clt2,cost = create_all(nodes1,nodes2,cost_file=cost_file)
    dp = DynProg(1,cost)
    dp.comp_nodes(clt1['root'],clt2['root'])
    cost_dict = dp.cost_dict
    bt_dict = dp.bt_dict
    ff = pd.Series(cost_dict).unstack().loc[clt1.keys(),clt2.keys()]
    ff2 = ff.loc[rr.index,rr.columns]
    bt_names = {k:v.name for k,v in bt_dict.items()}
    bt = pd.Series(bt_names).unstack().loc[ff2.index,ff2.columns]
    assert(allclose(rr,ff2))
    assert((bt_orig!=bt).sum().sum()==0)
    print('All Good!')
